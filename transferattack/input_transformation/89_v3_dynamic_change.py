import os
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
import weakref
from typing import List, Optional, Dict, Tuple

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class GradHookManager:
    """钩子管理器,使用 weakref 避免循环引用"""
    def __init__(self):
        self.hooks = []
        
    def register_hook(self, module: nn.Module, hook_fn):
        handle = module.register_backward_hook(hook_fn)
        self.hooks.append(weakref.ref(handle))
        
    def clear_hooks(self):
        for hook_ref in self.hooks:
            hook = hook_ref()
            if hook is not None:
                hook.remove()
        self.hooks.clear()
        
    def __del__(self):
        self.clear_hooks()

class SGMMixin:
    """改进的浅层梯度增强混入类"""
    def init_sgm(self, model: nn.Module, num_layers: int = 5):
        """增加捕获的浅层数量,并按深度分配权重"""
        self.hook_manager = GradHookManager()
        self.gradients: List[Tuple[torch.Tensor, float]] = []  # (gradient, weight)
        
        # 获取所有卷积层
        conv_layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
                
        # 选择前 num_layers 个卷积层,并分配权重
        selected_layers = conv_layers[:num_layers]
        weights = [1 - (i / num_layers) for i in range(num_layers)]  # 浅层权重更大
        
        for layer, weight in zip(selected_layers, weights):
            def hook_factory(weight):
                def hook(_, __, grad_output):
                    self.gradients.append((grad_output[0].detach(), weight))
                return hook
            
            self.hook_manager.register_hook(layer, hook_factory(weight))
            
    def get_sgm_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """改进的梯度融合方法"""
        if not self.gradients:
            return grad
            
        B, C, H, W = grad.shape
        combined_grad = grad.clone()
        
        for g, weight in self.gradients:
            # 1. 调整批次大小
            if g.size(0) != B:
                if g.size(0) > B:
                    # 使用相同的随机索引进行采样
                    idx = torch.randperm(g.size(0))[:B]
                    g = g[idx]
                else:
                    # 使用随机采样进行复制
                    repeat_factor = (B + g.size(0) - 1) // g.size(0)
                    g = g.repeat(repeat_factor, 1, 1, 1)[:B]
            
            # 2. 处理通道数
            if g.shape[1] != C:
                g = g.mean(dim=1, keepdim=True).repeat(1, C, 1, 1)
            
            # 3. 处理空间维度
            if g.shape[2:] != (H, W):
                g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
            
            # 4. 归一化并加权
            g = F.normalize(g, p=2, dim=(2,3))
            combined_grad = combined_grad + self.sgm_gamma * weight * g
        
        combined_grad = F.normalize(combined_grad, p=2, dim=(2,3))
        self.gradients.clear()
        return combined_grad

class GradCAMMixin:
    """新增的 Grad-CAM 混入类"""
    def __init__(self):
        self.hook_manager = GradHookManager()
        self.activations = {}
        self.gradients = {}
        
    def init_grad_cam(self, model: nn.Module):
        """预定义关键层而不是所有卷积层"""
        target_layers = []
        for name, module in model.named_modules():
            if any(layer in name for layer in ['layer1', 'layer2', 'layer3', 'layer4']):
                if isinstance(module, nn.Conv2d):
                    target_layers.append((name, module))
                    
        for name, module in target_layers:
            self.hook_manager.register_hook(
                module,
                lambda n: lambda m, i, o: setattr(self.activations, n, o),
                name
            )
            self.hook_manager.register_hook(
                module,
                lambda n: lambda m, gi, go: setattr(self.gradients, n, go[0]),
                name
            )
            
    def compute_saliency(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """优化的显著性图计算"""
        try:
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            loss = self.get_loss(logits, label)
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            
            saliency_maps = []
            # 动态权重计算
            layer_weights = {}
            total_weight = 0
            
            for name, act in self.activations.items():
                if name not in self.gradients:
                    continue
                    
                grad = self.gradients[name]
                # 计算显著性分数
                importance_score = grad.abs().mean()
                layer_weights[name] = float(importance_score)
                total_weight += layer_weights[name]
            
            # 归一化权重
            if total_weight > 0:
                for name in layer_weights:
                    layer_weights[name] /= total_weight
            
            for name, act in self.activations.items():
                if name not in self.gradients or name not in layer_weights:
                    continue
                    
                weight = layer_weights[name]
                grad = self.gradients[name]
                
                alpha = grad.mean(dim=(2, 3), keepdim=True)
                cam = (alpha * act).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                saliency_maps.append(weight * cam)
            
            if not saliency_maps:
                return torch.zeros_like(x[:,:1])
                
            return sum(saliency_maps)
            
        finally:
            self.activations.clear()
            self.gradients.clear()

class FreqPerturbationMixin:
    """新增的频域扰动混入类"""
    def create_freq_filters(self, h: int, w: int) -> Dict[str, torch.Tensor]:
        """创建不同类型的频域滤波器"""
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w)
        )
        D = torch.sqrt(X**2 + Y**2)
        
        # 高频滤波器
        high_pass = 1 - torch.exp(-D**2 / 0.2)
        
        # 低频滤波器
        low_pass = torch.exp(-D**2 / 0.2)
        
        # 中频带通滤波器
        band_pass = ((D >= 0.3) & (D <= 0.7)).float()
        
        return {
            'high': high_pass,
            'low': low_pass,
            'band': band_pass
        }
    
    def freq_perturbation(
        self,
        x: torch.Tensor,
        current_iter: int,
        total_iters: int
    ) -> torch.Tensor:
        """改进的频域扰动"""
        # 余弦衰减
        progress = current_iter / total_iters
        current_epsilon = self.freq_epsilon * (0.5 * (1 + np.cos(np.pi * progress)))
        
        filters = self.create_freq_filters(x.size(2), x.size(3))
        filters = {k: v.to(x.device) for k, v in filters.items()}
        
        # FFT
        Xf = torch.fft.rfft2(x)
        
        # 分别对不同频段进行扰动
        perturbations = []
        weights = {'high': 0.5, 'low': 0.3, 'band': 0.2}  # 不同频段权重
        
        for ftype, weight in weights.items():
            noise = torch.complex(
                torch.randn_like(Xf.real),
                torch.randn_like(Xf.imag)
            )
            filter_expanded = filters[ftype].unsqueeze(0).unsqueeze(0)
            pert = Xf + current_epsilon * weight * noise * filter_expanded
            perturbations.append(torch.fft.irfft2(pert))
            
        # 融合不同频段的扰动
        x_perturbed = sum(weights[ftype] * p for ftype, p in zip(weights.keys(), perturbations))
        return torch.clamp(x_perturbed, 0, 1)

class BSR(MIFGSM):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'(https://arxiv.org/abs/2308.10299)
    This class applies block-wise shuffle and random rotations to generate adversarial examples.
    """
    def __init__(
            self,
            model_name,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.,
            num_scale=20,
            num_block=3,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=None,
            attack='BSR',
            **kwargs
    ):
        super().__init__(model_name, epsilon, alpha, epoch, decay,
                         targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block

        # Random rotation within a specified degree range
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

    def get_length(self, length):
        """
        Randomly split a dimension (length) into num_block segments of uneven sizes.
        Returns a tuple with each segment length that sums to 'length'.
        """
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += (length - rand_norm.sum())
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        """
        Shuffle blocks along a single spatial dimension 'dim'.
        """
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        """
        Random rotation within ±24 degrees (using bilinear interpolation).
        """
        return self.rotation_transform(x)

    def shuffle(self, x):
        """
        Randomly shuffle blocks along width and height in randomized order.
        """
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])  # shuffle along the first chosen dim
        out_strips = []
        for x_strip in x_strips:
            # Apply random rotation before shuffling along the second dim
            x_rotated = self.image_rotation(x_strip)
            # Shuffle along the second dim
            out_strips.append(
                torch.cat(self.shuffle_single_dim(x_rotated, dims[1]), dim=dims[1])
            )
        return torch.cat(out_strips, dim=dims[0])

    def transform(self, x, **kwargs):
        """
        Standard BSR transform: replicate the shuffled images num_scale times.
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)], dim=0)

    def get_loss(self, logits, label):
        """
        Repeat label to match the expanded batch and calculate basic cross-entropy or targeted loss.
        此处注意，如果 logits 的 batch_size 与 label 本身相等，则不进行重复，
        避免在 Grad-CAM 计算时出现 batch_size 不匹配的错误。
        """
        if logits.size(0) == label.size(0):
            repeated_labels = label
        else:
            repeated_labels = label.repeat(self.num_scale)
        return -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)


class CrossBSR(BSR):
    """
    CrossBSR:
      Introduces cross-image block mixing before the standard BSR shuffle.
      mix_ratio: Probability of using another image's block.
    """
    def __init__(
            self,
            model_name,
            mix_ratio=0.05,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.,
            num_scale=20,
            num_block=3,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=None,
            attack='CrossBSR',
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            epsilon=epsilon,
            alpha=alpha,
            epoch=epoch,
            decay=decay,
            num_scale=num_scale,
            num_block=num_block,
            targeted=targeted,
            random_start=random_start,
            norm=norm,
            loss=loss,
            device=device,
            attack=attack,
            **kwargs
        )
        self.mix_ratio = mix_ratio

    def cross_shuffle(self, x):
        """
        For each image, pick another random image from the batch and
        merge their blocks in the spatial dimension with probability mix_ratio.
        """
        B, C, H, W = x.shape
        x_mixed = []
        split_h = self.get_length(H)

        for i in range(B):
            # Randomly choose another index j
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i : i+1]
            x_j = x[j : j+1]

            # Split along height
            strips_i = x_i.split(split_h, dim=2)
            strips_j = x_j.split(split_h, dim=2)

            # Build mixed strips
            mixed_strips = []
            for k in range(self.num_block):
                if random.random() < self.mix_ratio:
                    mixed_strips.append(strips_j[k])
                else:
                    mixed_strips.append(strips_i[k])
            x_mixed.append(torch.cat(mixed_strips, dim=2))

        return torch.cat(x_mixed, dim=0)

    def transform(self, x, **kwargs):
        """
        Apply cross-image mixing once, then do the standard BSR shuffle and replicate.
        """
        x_mixed = self.cross_shuffle(x)
        return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)


class EnhancedCrossBSR(CrossBSR, SGMMixin, GradCAMMixin, FreqPerturbationMixin):
    """
    增强版 CrossBSR 攻击，整合多策略：
      1. 动态混合比例（Adaptive Mix）
      2. 多尺度输入变换（DIM）
      3. 浅层梯度增强（SGM）
      4. 防御模拟（NRDM）
      5. Grad-CAM 显著性区域引导混合
      6. 频域扰动 (FFT-domain perturbation)

    本版本新增利用模型所有卷积层计算 Grad-CAM 显著性图，并采用动态权重融合，
    越靠后层占得权重越大。同时，在每轮迭代结束后将计算得到的显著性图保存到指定目录。
    """
    def __init__(
            self,
            # 基础参数
            model_name,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.,
            num_scale=20,
            num_block=2,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=None,
            attack='EnhancedCrossBSR',
            mix_ratio=0.05,
            # 扩展参数
            use_dim=True,
            dim_resize_range=(0.8, 1.2),
            use_adaptive_mix=True,
            mix_decay=0.95,
            use_sgm=True,
            sgm_gamma=0.5,
            use_defense_sim=False,
            defense_type='jpeg',
            # Grad-CAM 显著性相关
            use_saliency_map=True,
            saliency_weight=0.1,
            gradcam_target_layer=["layer1", "layer2"],  # 原有接口（可用于普通显著性融合）
            # 频域扰动
            use_freq_perturbation=True,
            freq_epsilon=0.05,  # 控制频域扰动强度
            # 显著性图保存目录
            saliency_save_dir='saliency_maps',
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            mix_ratio=mix_ratio,
            epsilon=epsilon,
            alpha=alpha,
            epoch=epoch,
            decay=decay,
            num_scale=num_scale,
            num_block=num_block,
            targeted=targeted,
            random_start=random_start,
            norm=norm,
            loss=loss,
            device=device,
            attack=attack,
            **kwargs
        )

        # 各模块开关/超参数
        self.use_dim = use_dim
        self.dim_resize_range = dim_resize_range
        self.use_adaptive_mix = use_adaptive_mix
        self.mix_decay = mix_decay
        self.use_sgm = use_sgm
        self.sgm_gamma = sgm_gamma
        self.use_defense_sim = use_defense_sim
        self.defense_type = defense_type
        self.use_saliency_map = use_saliency_map
        self.saliency_weight = saliency_weight
        self.gradcam_target_layer = gradcam_target_layer  # 原有接口
        self.use_freq_perturbation = use_freq_perturbation
        self.freq_epsilon = freq_epsilon

        # 显著性图保存目录（传入后，每轮迭代的显著性图会保存成图片）
        self.saliency_save_dir = saliency_save_dir
        if self.saliency_save_dir is not None:
            os.makedirs(self.saliency_save_dir, exist_ok=True)

        # 浅层梯度增强（SGM）
        if self.use_sgm:
            self.init_sgm(self.model)

        # Momentum buffer for some variants
        self.momentum = 0

    # -----------------------------
    # 1. 使用所有卷积层进行动态加权 Grad-CAM 显著性图计算
    # -----------------------------
    def grad_cam_saliency_dynamic(self, x, label):
        """
        计算模型所有卷积层的 Grad-CAM 显著性图，并采用动态权重融合：
          1) 对模型中所有 nn.Conv2d 层注册前向与反向 hook，捕获激活和梯度。
          2) 分别计算每个层的 Grad-CAM 显著性图 (alpha = GAP(梯度) 后与激活图相乘求和，并经过 ReLU)；
             然后上采样到与输入 x 相同分辨率并归一化。
          3) 根据层在列表中的顺序给予动态权重（越靠后权重越大），加权融合所有显著性图。
        """
        # 筛选模型中所有卷积层
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))

        activations_dict = {}
        gradients_dict = {}
        hooks = []

        # 定义 hook 工厂函数
        def forward_hook_factory(name):
            def forward_hook(module, inp, out):
                activations_dict[name] = out
            return forward_hook

        def backward_hook_factory(name):
            def backward_hook(module, grad_in, grad_out):
                gradients_dict[name] = grad_out[0]
            return backward_hook

        # 为所有卷积层注册 hook
        for name, module in conv_layers:
            handle_f = module.register_forward_hook(forward_hook_factory(name))
            handle_b = module.register_backward_hook(backward_hook_factory(name))
            hooks.append(handle_f)
            hooks.append(handle_b)

        # Forward & backward propagation
        x_ = x.clone().detach().requires_grad_(True)
        logits = self.model(x_)
        loss_val = self.get_loss(logits, label)
        self.model.zero_grad()
        loss_val.backward(retain_graph=True)

        # 移除 hook
        for h in hooks:
            h.remove()

        saliency_maps = []
        # 按 conv_layers 注册顺序计算 Grad-CAM
        for idx, (name, module) in enumerate(conv_layers):
            act = activations_dict.get(name, None)
            grad = gradients_dict.get(name, None)
            if act is None or grad is None:
                continue

            # 计算 alpha = 全局平均池化梯度
            alpha = grad.view(grad.size(0), grad.size(1), -1).mean(dim=2)
            alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1)
            weighted = alpha * act
            cam = weighted.sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam_up = F.interpolate(cam, size=x_.shape[-2:], mode='bilinear', align_corners=False)
            # 归一化
            cam_up = cam_up - cam_up.min()
            cam_up = cam_up / (cam_up.max() + 1e-8)
            saliency_maps.append(cam_up)

        if not saliency_maps:
            return torch.zeros_like(x_[:, :1])

        # 动态权重：根据层的位置分配权重，越靠后层（即 idx 越大）权重越大
        num_layers = len(saliency_maps)
        weights = torch.tensor([float(i+1) for i in range(num_layers)], device=x.device)
        weights = weights / weights.sum()  # 归一化权重

        # 加权融合所有显著性图
        fused_cam = 0
        for w, cam in zip(weights, saliency_maps):
            fused_cam += w * cam
        return fused_cam.detach()

    # -----------------------------
    # 2. Grad-CAM 多阶融合显著性图计算（原有接口）
    # -----------------------------
    def grad_cam_saliency(self, x, label, target_layers=None):
        """
        使用 Grad-CAM 计算多阶融合显著性图:
          1) 注册前向与反向钩子，捕获多个目标层的激活和梯度。
          2) 对每个目标层计算 alpha = GAP(梯度) 和 CAM = ReLU(激活 * alpha)。
          3) 将各层的 CAM 上采样到输入尺寸后融合（此处采用平均融合）。
        参数:
          x: 输入张量。
          label: 标签。
          target_layers: 目标层名称列表，如果为 None 则默认使用 ["layer1"]。
        """
        print(f"进行多阶融合显著性图计算，目标层: {target_layers}")
        if target_layers is None:
            target_layers = ["layer1"]

        activations_dict = {}
        gradients_dict = {}
        hooks = []

        def forward_hook_factory(name):
            def forward_hook(module, inp, out):
                activations_dict[name] = out
            return forward_hook

        def backward_hook_factory(name):
            def backward_hook(module, grad_in, grad_out):
                gradients_dict[name] = grad_out[0]
            return backward_hook

        for n, m in self.model.named_modules():
            for target_layer in target_layers:
                if target_layer in n:
                    handle_f = m.register_forward_hook(forward_hook_factory(n))
                    handle_b = m.register_backward_hook(backward_hook_factory(n))
                    hooks.append(handle_f)
                    hooks.append(handle_b)
                    break

        x_ = x.clone().detach().requires_grad_(True)
        logits = self.model(x_)
        loss_val = self.get_loss(logits, label)
        self.model.zero_grad()
        loss_val.backward(retain_graph=True)

        for h in hooks:
            h.remove()

        cam_list = []
        for layer_name in target_layers:
            act = activations_dict.get(layer_name, None)
            grad = gradients_dict.get(layer_name, None)
            if act is None or grad is None:
                continue

            alpha = grad.view(grad.size(0), grad.size(1), -1).mean(dim=2)
            alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1)
            weighted = alpha * act
            cam = weighted.sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam_up = F.interpolate(cam, size=x_.shape[-2:], mode='bilinear', align_corners=False)
            cam_up = cam_up - cam_up.min()
            cam_up = cam_up / (cam_up.max() + 1e-8)
            cam_list.append(cam_up)

        if not cam_list:
            return torch.zeros_like(x_[:, :1])
        fused_cam = torch.stack(cam_list, dim=0).mean(dim=0)
        return fused_cam.detach()

    # -----------------------------
    # 3. 频域扰动 (FFT-domain)
    # -----------------------------
    def freq_perturbation(self, x, current_iter):
        """
        在频域对图像添加随机扰动
        根据当前迭代次数采用动态衰减策略
        """
        if current_iter < 5:      # 前5次迭代快速衰减
            decay = 0.85
        elif current_iter < 15:   # 中期稳定衰减
            decay = 0.93
        else:                     # 后期慢速衰减
            decay = 0.97
        current_epsilon = self.freq_epsilon * (decay ** current_iter)

        Xf = torch.fft.fftn(x, dim=(-2, -1))
        noise_real = torch.randn_like(Xf.real)
        noise_imag = torch.randn_like(Xf.imag)

        Xf_real_perturbed = Xf.real + current_epsilon * noise_real
        Xf_imag_perturbed = Xf.imag + current_epsilon * noise_imag
        Xf_perturbed = torch.complex(Xf_real_perturbed, Xf_imag_perturbed)
        perturbed = torch.fft.ifftn(Xf_perturbed, dim=(-2, -1)).real
        perturbed = torch.clamp(perturbed, 0.0, 1.0)
        return perturbed

    # -----------------------------
    # 4. 基于显著性区域的跨图像混合
    # -----------------------------
    def region_based_cross_shuffle(self, x, label):
        B, C, H, W = x.shape
        x_mixed = []
        print('即将开始计算显著性图')
        # 这里选择使用动态权重融合的 Grad-CAM 显著性图
        saliency = self.grad_cam_saliency_dynamic(x, label)
        saliency = saliency.clamp(min=0., max=1.)

        split_h = self.get_length(H)
        for i in range(B):
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i : i+1]
            x_j = x[j : j+1]
            s_i = saliency[i : i+1]
            s_j = saliency[j : j+1]

            strips_i = x_i.split(split_h, dim=2)
            strips_j = x_j.split(split_h, dim=2)
            s_strips_i = s_i.split(split_h, dim=2)
            s_strips_j = s_j.split(split_h, dim=2)

            mixed_strips = []
            for k in range(self.num_block):
                mean_sal_i = s_strips_i[k].mean().item()
                mean_sal_j = s_strips_j[k].mean().item()
                if (random.random() < self.mix_ratio) or (mean_sal_i > mean_sal_j):
                    mixed_strips.append(strips_j[k])
                else:
                    mixed_strips.append(strips_i[k])
            x_mixed.append(torch.cat(mixed_strips, dim=2))
        return torch.cat(x_mixed, dim=0)

    # -----------------------------
    # 5. 多尺度变换 (DIM)
    # -----------------------------
    def dim_transform(self, x):
        resize_factor = random.uniform(*self.dim_resize_range)
        original_h, original_w = x.size(2), x.size(3)
        new_h = int(original_h * resize_factor)
        new_w = int(original_w * resize_factor)

        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

        def calculate_pad_crop(orig_size, new_size):
            delta = orig_size - new_size
            if delta >= 0:
                pad_front = random.randint(0, delta)
                pad_back = delta - pad_front
                return pad_front, pad_back
            else:
                crop_size = -delta
                crop_start = random.randint(0, crop_size)
                return -crop_start, -(crop_size - crop_start)
        pad_top, pad_bottom = calculate_pad_crop(original_h, new_h)
        pad_left, pad_right = calculate_pad_crop(original_w, new_w)
        return F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))

    # -----------------------------
    # 6. 防御模拟
    # -----------------------------
    def defense_emulate(self, x):
        if self.defense_type == 'jpeg':
            return self.jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def jpeg_compress(self, x, quality=75):
        # 仅作示例，不做实际 JPEG 算法实现
        return x

    # -----------------------------
    # 7. 变换流程整合
    # -----------------------------
    def update_mix_ratio(self):
        if self.use_adaptive_mix:
            self.mix_ratio = max(0.01, self.mix_ratio * self.mix_decay)

    def transform(self, x, label=None, saliency_map=None, momentum=None, current_iter=0, **kwargs):
        """
        1. 多尺度变换 (DIM)
        2. 显著性区域混合或普通跨图像混合
        3. 防御模拟
        4. 频域扰动
        5. BSR 标准 block shuffle 并 replicate num_scale 次
        """
        _x = x
        if self.use_dim:
            _x = self.dim_transform(_x)

        # 1. 先进行基本的跨图像混合
        if self.use_saliency_map and (label is not None):
            _x = self.region_based_cross_shuffle(_x, label)
        else:
            _x = self.cross_shuffle(_x)

        # 2. 应用防御模拟
        if self.use_defense_sim:
            _x = self.defense_emulate(_x)

        # 3. 添加频域扰动
        if self.use_freq_perturbation:
            _x = self.freq_perturbation(_x, current_iter=current_iter)

        # 4. 执行 BSR shuffle 并复制
        shuffled = []
        for _ in range(self.num_scale):
            shuffled.append(self.shuffle(_x))
        
        # 5. 确保所有批次大小一致
        result = torch.cat(shuffled, dim=0)
        if result.size(0) != x.size(0) * self.num_scale:
            result = result[:x.size(0) * self.num_scale]
            
        return result

    # -----------------------------
    # 8. 显著性区域损失
    # -----------------------------
    def saliency_region_loss(self, x_adv, x_orig, label):
        with torch.no_grad():
            cam_orig = self.grad_cam_saliency_dynamic(x_orig, label)
        diff = (x_adv - x_orig).abs().mean(dim=1, keepdim=True)
        diff_sal = diff * cam_orig
        return diff_sal.mean()

    # -----------------------------
    # 9. 综合损失
    # -----------------------------
    def get_loss(self, logits, label, x_adv=None, x_orig=None):
        """
        对 logits 和 label 进行匹配：如果 logits 的 batch_size 与 label 不一致，则重复 label。
        """
        if isinstance(label, int):
            label = torch.tensor([label], dtype=torch.long, device=logits.device)
        if logits.size(0) == label.size(0):
            repeated_labels = label
        else:
            repeated_labels = label.repeat(self.num_scale)
        base_loss = -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)

        if self.use_saliency_map and (x_adv is not None) and (x_orig is not None):
            sal_loss = self.saliency_region_loss(x_adv, x_orig, label)
            total_loss = base_loss + self.saliency_weight * sal_loss
        else:
            total_loss = base_loss
        return total_loss

    # -----------------------------
    # 10. 梯度计算 (整合 SGM)
    # -----------------------------
    def get_grad(self, loss, delta_or_x):
        grad = super(CrossBSR, self).get_grad(loss, delta_or_x)
        return self.get_sgm_grad(grad) if self.use_sgm else grad

    # -----------------------------
    # 11. 统一的 forward 函数
    # -----------------------------
    def forward(self, data, label, **kwargs):
        """
        通用攻击流程：
         - 如果 self.targeted 为 True，则 label 必须包含两个元素（真实标签和目标标签），取第二个作为目标标签；
         - 否则 label 为单一标签。
         - 使用 momentum 缓冲计算梯度，并更新 delta。
         同时，每轮迭代结束后保存一次本轮计算的显著性图（如果设置了 saliency_save_dir）。
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        #print(f"开始攻击，标签: {label.item()}")
        # 初始化 delta
        delta = self.init_delta(data)
        self.momentum = 0

        for i in tqdm(range(self.epoch), desc=f"Attack: {self.attack} attack_model: {self.model_name}"):
            transformed = self.transform(data + delta, label=label, current_iter=i)
            logits = self.model(transformed)
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            self.momentum = self.get_momentum(grad, self.momentum)
            delta = self.update_delta(delta, data, self.momentum, self.alpha)

            # 保存当前迭代的显著性图
            if self.saliency_save_dir is not None:
                # 重新计算显著性图（或者直接采用 region_based_cross_shuffle 内计算的 saliency，
                # 这里采用动态融合的显著性图计算）
                sal_map = self.grad_cam_saliency_dynamic(data + delta, label)
                save_path = os.path.join(self.saliency_save_dir, f"saliency_iter_{i}.png")
                save_image(sal_map, save_path)
        return delta.detach()

    def __del__(self):
        """
        清理 hook (handles)
        """
        for handle in getattr(self, 'handles', []):
            handle.remove()