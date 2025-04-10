import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import *
from ..gradient.mifgsm import MIFGSM

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
        """
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


class EnhancedCrossBSR(CrossBSR):
    """
    增强版 CrossBSR 攻击，整合多策略：
      1. 动态混合比例（Adaptive Mix）
      2. 多尺度输入变换（DIM）
      3. 浅层梯度增强（SGM）
      4. 防御模拟（NRDM）
      5. Grad-CAM 显著性区域引导混合
      6. 频域扰动 (FFT-domain perturbation)
    """
    def __init__(
            self,
            # 基础
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
            # 扩展
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
            gradcam_target_layer="layer1",  # 指定要在何处计算Grad-CAM
            # 频域扰动
            use_freq_perturbation=True,
            freq_epsilon=0.05,  # 控制频域扰动强度
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
        self.gradcam_target_layer = gradcam_target_layer

        self.use_freq_perturbation = use_freq_perturbation
        self.freq_epsilon = freq_epsilon

        # 浅层梯度增强（SGM）
        if self.use_sgm:
            self._init_sgm_hooks()

        # Momentum buffer for some variants (like MIFGSM-based)
        self.momentum = 0

    # -----------------------------
    # 1. 浅层梯度增强 (SGM)
    # -----------------------------
    def _init_sgm_hooks(self):
        """优化SGM钩子管理"""
        self.gradients = {}
        self.handles = []
        
        # 定义要监控的层
        target_layers = ['layer1', 'layer2'] 
        
        for name, module in self.model.named_modules():
            if any(layer in name for layer in target_layers):
                handle = module.register_backward_hook(
                    lambda n, gi, go, name=name: self._save_grad(name, gi, go))
                self.handles.append(handle)
    
    def _save_grad(self, name, grad_input, grad_output):
        """改进梯度存储逻辑"""
        if grad_output[0] is not None:
            self.gradients[name] = grad_output[0].detach()

    # -----------------------------
    # 2. Grad-CAM 显著性图计算
    # -----------------------------
    def grad_cam_saliency(self, x, label, target_layer="layer1"):
        """
        使用 Grad-CAM 计算显著性图:
          1) 注册前向/反向钩子，捕获中间输出(activation)和梯度
          2) 计算 alpha = GAP(梯度)
          3) cam = Relu(activation * alpha)
        """
        activations = {}
        gradients = {}

        def forward_hook(module, inp, out):
            activations['value'] = out

        def backward_hook(module, grad_in, grad_out):
            gradients['value'] = grad_out[0]

        chosen_handle_f, chosen_handle_b = None, None
        for n, m in self.model.named_modules():
            if target_layer in n:
                chosen_handle_f = m.register_forward_hook(forward_hook)
                chosen_handle_b = m.register_backward_hook(backward_hook)
                break

        x_ = x.clone().detach().requires_grad_(True)
        logits = self.model(x_)
        loss_val = self.get_loss(logits, label)
        self.model.zero_grad()
        loss_val.backward(retain_graph=True)

        if chosen_handle_f is not None:
            chosen_handle_f.remove()
        if chosen_handle_b is not None:
            chosen_handle_b.remove()

        act = activations.get('value', None)
        grad = gradients.get('value', None)
        if act is None or grad is None:
            return torch.zeros_like(x_[:, :1])

        alpha = grad.view(grad.size(0), grad.size(1), -1).mean(dim=2)
        alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1)

        weighted = alpha * act
        cam = weighted.sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_up = F.interpolate(cam, size=x_.shape[-2:], mode='bilinear', align_corners=False)
        cam_up = cam_up - cam_up.min()
        cam_up = cam_up / (cam_up.max() + 1e-8)
        return cam_up.detach()

    # -----------------------------
    # 3. 频域扰动 (FFT-domain)
    # -----------------------------
    def freq_perturbation(self, x):
        """增强频域扰动"""
        # FFT变换
        Xf = torch.fft.fftn(x, dim=(-2, -1))
        
        # 生成高频和低频噪声
        noise_high = torch.randn_like(Xf) * self.freq_epsilon
        noise_low = F.avg_pool2d(
            torch.randn_like(x), kernel_size=3, stride=1, padding=1
        ).repeat(1, 1, 1, 1)
        noise_low = torch.fft.fftn(noise_low, dim=(-2, -1))
        
        # 混合扰动
        mix_ratio = random.random()  # 动态调整高低频比例
        Xf_perturbed = Xf + mix_ratio * noise_high + (1 - mix_ratio) * noise_low
        
        # 反变换
        perturbed = torch.fft.ifftn(Xf_perturbed, dim=(-2, -1)).real
        return torch.clamp(perturbed, 0.0, 1.0)

    # -----------------------------
    # 4. 基于显著性区域的跨图像混合
    # -----------------------------
    def region_based_cross_shuffle(self, x, label):
        """改进区域混合策略"""
        B, C, H, W = x.shape
        x_mixed = []
        
        # 计算显著性图
        with torch.no_grad():
            saliency = self.grad_cam_saliency(x, label)
        
        # 同时在高度和宽度方向分块
        split_h = self.get_length(H)
        split_w = self.get_length(W)
        
        for i in range(B):
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i:i+1]
            x_j = x[j:j+1]
            s_i = saliency[i:i+1]
            s_j = saliency[j:j+1]
            
            # 垂直分块
            strips_i_v = x_i.split(split_h, dim=2)
            strips_j_v = x_j.split(split_h, dim=2)
            s_strips_i_v = s_i.split(split_h, dim=2)
            s_strips_j_v = s_j.split(split_h, dim=2)
            
            # 水平分块并混合
            mixed_strips = []
            for k, (strip_i, strip_j, s_i_v, s_j_v) in enumerate(zip(
                strips_i_v, strips_j_v, s_strips_i_v, s_strips_j_v)):
                
                # 计算显著性得分
                sal_score_i = s_i_v.mean().item()
                sal_score_j = s_j_v.mean().item()
                
                # 基于显著性和随机性决定混合
                if (random.random() < self.mix_ratio) or (sal_score_i > sal_score_j):
                    mixed_strips.append(strip_j)
                else:
                    mixed_strips.append(strip_i)
                    
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

    def transform(self, x, label=None, **kwargs):
        """优化变换流程"""
        batch_size = x.size(0)
        
        # 1. 多尺度变换
        if self.use_dim:
            x = self.dim_transform(x)
            
        # 2. 区域混合
        if self.use_saliency_map and label is not None:
            x = self.region_based_cross_shuffle(x, label)
        else:
            x = self.cross_shuffle(x)
            
        # 3. 频域扰动
        if self.use_freq_perturbation:
            x = self.freq_perturbation(x)
            
        # 4. 防御模拟
        if self.use_defense_sim:
            x = self.defense_emulate(x)
            
        # 5. BSR变换并确保批次一致性
        transformed = []
        for _ in range(self.num_scale):
            t = self.shuffle(x)
            if t.size(0) != batch_size:
                t = t[:batch_size]
            transformed.append(t)
            
        return torch.cat(transformed, dim=0)

    # -----------------------------
    # 8. 显著性区域损失
    # -----------------------------
    def saliency_region_loss(self, x_adv, x_orig, label):
        with torch.no_grad():
            cam_orig = self.grad_cam_saliency(x_orig, label, target_layer=self.gradcam_target_layer)
        diff = (x_adv - x_orig).abs().mean(dim=1, keepdim=True)
        diff_sal = diff * cam_orig
        return diff_sal.mean()

    # -----------------------------
    # 9. 综合损失
    # -----------------------------
    def get_loss(self, logits, label, x_adv=None, x_orig=None):
        if isinstance(label, int):
            label = torch.tensor([label], dtype=torch.long, device=logits.device)

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
    def get_grad(self, loss, delta):
        """改进梯度融合策略"""
        grad = super(CrossBSR, self).get_grad(loss, delta)
        
        if self.use_sgm and self.gradients:
            # 融合多层梯度
            shallow_grad = None
            for name, g in self.gradients.items():
                # 调整批次大小
                batch_size = delta.size(0)
                g = g.view(batch_size, self.num_scale, *g.shape[1:]).mean(dim=1)
                
                # 通道调整
                if g.size(1) > 3:
                    g = g.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                
                # 尺寸调整
                g = F.interpolate(g, size=grad.shape[-2:], 
                                mode='bilinear', align_corners=False)
                
                if shallow_grad is None:
                    shallow_grad = g
                else:
                    shallow_grad = shallow_grad + g
            
            # 加权融合
            grad += self.sgm_gamma * shallow_grad / len(self.gradients)
            self.gradients.clear()
            
        return grad

    # -----------------------------
    # 11. 统一的 forward 函数
    # -----------------------------
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure adapted according to the snippet:
         - If self.targeted is True, label must have length 2 [ground_truth_label, target_label]
         - Otherwise, label is just single dimensional
         - Use momentum buffer if needed
         - Transform, compute gradients, update delta
        """
        # Handle targeted label
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # second element is the targeted label

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize delta
        delta = self.init_delta(data)

        self.momentum = 0
        for _ in tqdm(range(self.epoch), desc=f"Attack: {self.attack} attack_model: {self.model_name} 当前正在结合多级sgm进行尝试----" ):
            # Example conditions (SIA, SIAWithSaliency) are placeholders
            # or could be replaced with your actual attack logic checks

            # Default transform
            logits = self.model(self.transform(data + delta, momentum=self.momentum))

            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)

            # Update momentum
            self.momentum = self.get_momentum(grad, self.momentum)
            # Update delta
            delta = self.update_delta(delta, data, self.momentum, self.alpha)

        return delta.detach()

    def __del__(self):
        """
        清理钩子 (handles)
        """
        for handle in getattr(self, 'handles', []):
            handle.remove()