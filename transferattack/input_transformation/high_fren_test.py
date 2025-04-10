# -*- coding: utf-8 -*-
"""
增强版 CrossBSR 攻击类
结合了CNN在频域对高频成分的敏感性，在图像频域内注入高频噪声，
并在原有的多尺度变换、随机旋转以及跨图像混合的基础上增加了一些
针对CNN结构特性的改进建议，比如在特征通道上进行随机混合（可进一步扩展）。
"""

import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class EnhancedCrossBSR(MIFGSM):
    """
    增强版 CrossBSR 攻击类，示例修正：
      - 移除了 with torch.no_grad() 对 grad_cam_saliency 的调用，确保反向传播时能够正常构建计算图。
      - 增加了在频域内加入高频噪声的操作，利用 FFT 对图像在高频区域加入随机噪声，进一步扰动输入。
      - 其他流程与此前版本一致，可以根据 CNN 结构特性进一步改进，例如对特征通道进行随机混合（可扩展）。
    """
    def __init__(
            self,
            model_name,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.,
            num_scale=20,
            num_block=2,
            targeted=False,
            random_start=True,
            norm='linfty',
            loss='crossentropy',
            device=None,
            attack='EnhancedCrossBSR',
            mix_ratio=0.05,
            # DIM 多尺度变换参数
            use_dim=True,
            dim_resize_range=(0.8, 1.2),
            # Adaptive mix 自适应混合参数
            use_adaptive_mix=True,
            mix_decay=0.95,
            # SGM 参数
            use_sgm=True,
            sgm_gamma=0.5,
            # 防御模拟参数
            use_defense_sim=False,
            defense_type='jpeg',
            # 显著性图参数
            use_saliency_map=True,
            saliency_weight=0.1,
            gradcam_target_layer="layer1",
            # 频域扰动参数
            use_freq_perturbation=False,
            freq_epsilon=0.05,
            # integrated grad 参数
            use_integrated_grad=False,
            ig_steps=50,
            # 高频噪声参数（新增）
            use_high_freq_noise=True,
            high_freq_noise_level=0.1,
            high_freq_threshold=0.2,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            epsilon=epsilon,
            alpha=alpha,
            epoch=epoch,
            decay=decay,
            targeted=targeted,
            random_start=random_start,
            norm=norm,
            loss=loss,
            device=device,
            attack=attack
        )
        self.num_scale = num_scale
        self.num_block = num_block
        self.mix_ratio = mix_ratio

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
        self.gradcam_target_layer = gradcam_target_layer

        self.use_freq_perturbation = use_freq_perturbation
        self.freq_epsilon = freq_epsilon

        self.use_integrated_grad = use_integrated_grad
        self.ig_steps = ig_steps

        # 新增高频噪声参数
        self.use_high_freq_noise = use_high_freq_noise
        self.high_freq_noise_level = high_freq_noise_level
        self.high_freq_threshold = high_freq_threshold

        # 随机旋转变换
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

        if self.use_sgm:
            self._init_sgm_hooks()

        self.momentum = 0  # MIFGSM的 momentum 缓冲

    def _init_sgm_hooks(self):
        """初始化 SGM 相关的 backward hook。"""
        self.gradients = []
        self.handles = []
        for name, module in self.model.named_modules():
            if 'layer1' in name:
                handle = module.register_backward_hook(self._save_grad)
                self.handles.append(handle)
                break

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    # ------------------------- 工具函数 -------------------------
    def get_length(self, length):
        """将指定维度随机切分为 num_block 段，返回每一段的长度。"""
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += (length - rand_norm.sum())
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        """对指定 dim 下的 block 进行洗牌。"""
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        """对 x 进行一次随机旋转。"""
        return self.rotation_transform(x)

    # ------------------------- 显著性图计算 -------------------------
    def grad_cam_saliency(self, x, label, target_layer="layer1"):
        """
        使用 Grad-CAM 计算显著性图:
          1) 注册前向和反向钩子；
          2) 前向传播获取激活；
          3) 反向传播获取梯度；
          4) 计算 cam = ReLU(激活 * 全局平均池化(梯度))；
        """
        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            activations['value'] = out

        def backward_hook(module, grad_in, grad_out):
            gradients['value'] = grad_out[0]

        chosen_forward_hook, chosen_backward_hook = None, None
        for n, m in self.model.named_modules():
            if target_layer in n:
                chosen_forward_hook = m.register_forward_hook(forward_hook)
                chosen_backward_hook = m.register_backward_hook(backward_hook)
                break

        x_ = x.clone().detach().requires_grad_(True)
        logits = self.model(x_)
        base_loss = -self.loss(logits, label) if self.targeted else self.loss(logits, label)
        self.model.zero_grad()
        base_loss.backward(retain_graph=True)

        if chosen_forward_hook is not None:
            chosen_forward_hook.remove()
        if chosen_backward_hook is not None:
            chosen_backward_hook.remove()

        act = activations.get('value', None)
        grad = gradients.get('value', None)
        if act is None or grad is None:
            return torch.zeros_like(x_[:, :1])

        alpha = grad.view(grad.size(0), grad.size(1), -1).mean(dim=2)
        alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1)
        weighted = alpha * act
        cam = F.relu(weighted.sum(dim=1, keepdim=True))
        cam_up = F.interpolate(cam, size=x_.shape[-2:], mode='bilinear', align_corners=False)
        cam_up = cam_up - cam_up.min()
        cam_up = cam_up / (cam_up.max() + 1e-8)
        return cam_up.detach()

    def integrated_grad_saliency(self, x, label, steps=50):
        """
        使用积分梯度计算显著性图：
          baseline 为 0，
          IG = (x - baseline) * 累加(梯度) / steps
        """
        baseline = torch.zeros_like(x)
        x_diff = x - baseline
        x.requires_grad_(True)
        ig = torch.zeros_like(x)

        for alpha in torch.linspace(0, 1, steps):
            x_interpolated = baseline + alpha * x_diff
            x_interpolated = x_interpolated.clone().detach().requires_grad_(True)
            logits = self.model(x_interpolated)
            base_loss = -self.loss(logits, label) if self.targeted else self.loss(logits, label)
            self.model.zero_grad()
            base_loss.backward(retain_graph=False)
            if x_interpolated.grad is not None:
                ig += x_interpolated.grad

        ig = x_diff * ig / steps
        ig_sal = ig.abs().mean(dim=1, keepdim=True)
        ig_sal = ig_sal - ig_sal.min()
        ig_sal = ig_sal / (ig_sal.max() + 1e-8)
        x.requires_grad_(False)
        return ig_sal

    # ------------------------- 图像变换与混合 -------------------------
    def cross_shuffle(self, x):
        """跨图像混合：每个图像随机选取另一图像的部分 block 进行混合。"""
        B, C, H, W = x.shape
        x_mixed = []
        split_h = self.get_length(H)
        for i in range(B):
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i : i+1]
            x_j = x[j : j+1]
            strips_i = x_i.split(split_h, dim=2)
            strips_j = x_j.split(split_h, dim=2)
            mixed_strips = []
            for k in range(self.num_block):
                if random.random() < self.mix_ratio:
                    mixed_strips.append(strips_j[k])
                else:
                    mixed_strips.append(strips_i[k])
            x_mixed.append(torch.cat(mixed_strips, dim=2))
        return torch.cat(x_mixed, dim=0)

    def region_based_cross_shuffle(self, x, label):
        """
        基于显著性区域引导的跨图像混合：
          使用 Grad-CAM 或积分梯度计算显著性，再根据显著性均值选择对应 block。
        """
        B, C, H, W = x.shape
        if self.use_integrated_grad:
            print('使用了积分梯度')
            saliency = self.integrated_grad_saliency(x, label, steps=self.ig_steps)
        else:
            saliency = self.grad_cam_saliency(x, label, target_layer=self.gradcam_target_layer)
        saliency = saliency.clamp(0.0, 1.0)
        x_mixed = []
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

    def dim_transform(self, x):
        """多尺度变换 (DIM)：随机 resize 后 pad 或 crop 回原尺寸。"""
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

    def shuffle(self, x):
        """
        实现 BSR 中 block shuffle 加随机旋转。
        """
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        out_strips = []
        for x_strip in x_strips:
            x_rotated = self.image_rotation(x_strip)
            out_strips.append(torch.cat(self.shuffle_single_dim(x_rotated, dims[1]), dim=dims[1]))
        return torch.cat(out_strips, dim=dims[0])

    def defense_emulate(self, x):
        """防御模拟：例如 JPEG 压缩或添加高斯噪声。"""
        if self.defense_type == 'jpeg':
            return self.jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def jpeg_compress(self, x, quality=75):
        # 这里只作示例，不做具体 JPEG 压缩实现
        return x

    def freq_perturbation(self, x):
        """频域扰动：对 FFT 后的图像加上随机噪声，再反 FFT 获得扰动图像。"""
        Xf = torch.fft.fftn(x, dim=(-2, -1))
        noise_real = torch.randn_like(Xf.real)
        noise_imag = torch.randn_like(Xf.imag)
        Xf_real_perturbed = Xf.real + self.freq_epsilon * noise_real
        Xf_imag_perturbed = Xf.imag + self.freq_epsilon * noise_imag
        Xf_perturbed = torch.complex(Xf_real_perturbed, Xf_imag_perturbed)
        perturbed = torch.fft.ifftn(Xf_perturbed, dim=(-2, -1)).real
        return torch.clamp(perturbed, 0.0, 1.0)

    def add_high_freq_noise(self, x):
        """
        高频噪声注入：
          对 x 进行 FFT，将频域中高于阈值部分混入随机噪声，再反 FFT 得到图像。
        """
        # 使用二维 FFT
        Xf = torch.fft.fft2(x, norm='ortho')
        B, C, H, W = x.shape
        # 构造频率网格（单位归一化到 [0, 0.5]）
        freq_y = torch.linspace(0, 0.5, H, device=x.device).view(H, 1).expand(H, W)
        freq_x = torch.linspace(0, 0.5, W, device=x.device).view(1, W).expand(H, W)
        freq_magnitude = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        # 构造高频区域 mask，当频率大于阈值时 mask 值为 1
        mask = (freq_magnitude > self.high_freq_threshold).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # 形状 (1, 1, H, W)
        # 生成随机噪声并只作用于高频区域
        noise = self.high_freq_noise_level * torch.randn_like(Xf)
        Xf_noisy = Xf + noise * mask
        x_noisy = torch.fft.ifft2(Xf_noisy, norm='ortho').real
        x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
        return x_noisy

    def transform(self, x, label=None, momentum=None, **kwargs):
        """
        核心 transform 流程：
          1. 多尺度变换 (DIM)
          2. 显著性区域混合或普通 cross_shuffle
          3. 防御模拟
          4. 频域扰动
          5. 高频噪声注入（利用 CNN 对高频敏感性）
          6. BSR 标准 shuffle 并复制 num_scale 次
        """
        _x = x
        if self.use_dim:
            _x = self.dim_transform(_x)

        if self.use_saliency_map and (label is not None):
            _x = self.region_based_cross_shuffle(_x, label)
        else:
            _x = self.cross_shuffle(_x)

        if self.use_defense_sim:
            _x = self.defense_emulate(_x)

        if self.use_freq_perturbation:
            _x = self.freq_perturbation(_x)

        if self.use_high_freq_noise:
            _x = self.add_high_freq_noise(_x)

        return torch.cat([self.shuffle(_x) for _ in range(self.num_scale)], dim=0)

    # ------------------------- 损失函数与训练逻辑 -------------------------
    def saliency_region_loss(self, x_adv, x_orig, label):
        """
        计算显著性区域的额外损失约束，
        注意不使用 with torch.no_grad()，以免禁止梯度追踪。
        """
        if self.use_integrated_grad:
            cam_orig = self.integrated_grad_saliency(x_orig, label, steps=self.ig_steps)
        else:
            cam_orig = self.grad_cam_saliency(x_orig, label, target_layer=self.gradcam_target_layer)
        diff = (x_adv - x_orig).abs().mean(dim=1, keepdim=True)
        diff_sal = diff * cam_orig
        return diff_sal.mean()

    def get_loss(self, logits, label, x_adv=None, x_orig=None, is_for_saliency=False):
        """
        根据标定标签和对抗样本计算损失；
        若 is_for_saliency=True，则仅用于显著性计算，不重复标签。
        """
        if isinstance(label, int):
            label = torch.tensor([label], dtype=torch.long, device=logits.device)

        if not is_for_saliency:
            repeated_labels = label.repeat(self.num_scale)
            base_loss = -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)
        else:
            base_loss = -self.loss(logits, label) if self.targeted else self.loss(logits, label)

        if self.use_saliency_map and (x_adv is not None) and (x_orig is not None) and (not is_for_saliency):
            sal_loss = self.saliency_region_loss(x_adv, x_orig, label)
            total_loss = base_loss + self.saliency_weight * sal_loss
        else:
            total_loss = base_loss
        return total_loss

    def get_grad(self, loss, delta_or_x):
        """整合 SGM 梯度计算，加入浅层梯度校正。"""
        grad = super().get_grad(loss, delta_or_x)
        if self.use_sgm and len(self.gradients) > 0:
            shallow_grad = self.gradients[-1]
            batch_size = delta_or_x.size(0)
            shallow_grad = shallow_grad.view(batch_size, *shallow_grad.shape[1:])
            if shallow_grad.size(1) > 3:
                shallow_grad = shallow_grad.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            shallow_grad = F.interpolate(
                shallow_grad,
                size=grad.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            grad += self.sgm_gamma * shallow_grad
            self.gradients = []
        return grad

    def forward(self, data, label, **kwargs):
        """对抗攻击的核心流程。"""
        if self.targeted:
            # 定向攻击时 label 为 [原标签, 目标标签]，取目标标签
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        self.momentum = 0
        for _ in tqdm(range(self.epoch), desc=f"Attack: {self.attack}"):
            adv_data = data + delta
            adv_trans = self.transform(adv_data, label=label, momentum=self.momentum)
            logits = self.model(adv_trans)
            total_loss = self.get_loss(
                logits, label, x_adv=adv_data, x_orig=data, is_for_saliency=False
            )
            grad = self.get_grad(total_loss, delta)
            self.momentum = self.get_momentum(grad, self.momentum)
            delta = self.update_delta(delta, data, self.momentum, self.alpha)
            if self.use_adaptive_mix:
                self.mix_ratio = max(0.01, self.mix_ratio * self.mix_decay)
        return delta.detach()

    def __del__(self):
        """清理注册的 hook。"""
        for handle in getattr(self, 'handles', []):
            handle.remove()