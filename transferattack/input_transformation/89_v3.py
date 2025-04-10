import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class BSR(MIFGSM):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'
    (https://arxiv.org/abs/2308.10299)
    """
    def __init__(
            self,
            model_name,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.0,
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
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

    def get_length(self, length):
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += (length - rand_norm.sum())
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        return self.rotation_transform(x)

    def shuffle(self, x):
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        out_strips = []
        for x_strip in x_strips:
            x_rotated = self.image_rotation(x_strip)
            out_strips.append(
                torch.cat(self.shuffle_single_dim(x_rotated, dims[1]), dim=dims[1])
            )
        return torch.cat(out_strips, dim=dims[0])

    def transform(self, x, **kwargs):
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)], dim=0)

    def get_loss(self, logits, label):
        # Handle special cases like Inception where output is a tuple.
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.size(0) == label.size(0):
            repeated_labels = label
        else:
            repeated_labels = label.repeat(self.num_scale)
        return -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)


class CrossBSR(BSR):
    """
    CrossBSR Attack.
    Introduces cross-image block mixing before the standard BSR shuffle.
    """
    def __init__(
            self,
            model_name,
            mix_ratio=0.05,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.0,
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

    def transform(self, x, **kwargs):
        x_mixed = self.cross_shuffle(x)
        # Use explicit parent's shuffle function to avoid super() issues inside list comprehension.
        shuffle_func = super(CrossBSR, self).shuffle
        return torch.cat([shuffle_func(x_mixed) for _ in range(self.num_scale)], dim=0)


class EnhancedCrossBSR(CrossBSR):
    """
    EnhancedCrossBSR Attack.
    Integrates multiple strategies:
      1) Dynamic mix ratio (Adaptive Mix)
      2) DIM (multi-scale transform)
      3) Shallow Gradient Mining (SGM)
      4) Defense Simulation (NRDM)
      5) Grad-CAM saliency region guided mixing
      6) Frequency-domain perturbation (FFT)
    """
    def __init__(
            self,
            model_name,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.0,
            num_scale=20,
            num_block=2,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=None,
            attack='EnhancedCrossBSR',
            mix_ratio=0.05,
            use_dim=True,
            dim_resize_range=(0.8, 1.2),
            use_adaptive_mix=True,
            mix_decay=0.95,
            use_sgm=True,
            sgm_gamma=0.5,
            use_defense_sim=False,
            defense_type='jpeg',
            use_saliency_map=True,
            saliency_weight=0.1,
            gradcam_target_layer=["layer1", "layer2"],
            use_freq_perturbation=True,
            freq_epsilon=0.05,
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
        self.saliency_save_dir = saliency_save_dir
        if self.saliency_save_dir is not None:
            os.makedirs(self.saliency_save_dir, exist_ok=True)

        if self.use_sgm:
            self._init_sgm_hooks()

        self.momentum = 0

    def _init_sgm_hooks(self):
        self.gradients = []
        self.handles = []
        def forward_hook(module, inp, out):
            pass
        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0].detach())
        for name, module in self.model.named_modules():
            if 'layer1' in name and isinstance(module, nn.Module):
                h1 = module.register_forward_hook(forward_hook)
                h2 = module.register_backward_hook(backward_hook)
                self.handles.extend([h1, h2])
                break

    def get_length(self, length):
        return super().get_length(length)

    # --------------------------------------------------------------------------
    # 修正后的 Grad-CAM 显著性图计算函数
    # --------------------------------------------------------------------------
    def compute_saliency_map(self, x, label):
        """
        计算 Grad-CAM 显著性图：
         1. 开启梯度计算（with torch.enable_grad()）
         2. 强制将 x 的克隆版本设为 requires_grad_(True)
         3. 如果得到的 logits 为 tuple，则取 logits[0]
        """
        was_training = self.model.training
        with torch.enable_grad():
            self.model.train()  # 切换到 train 模式确保梯度计算
            x_ = x.clone().detach().requires_grad_(True)
            logits = self.model(x_)
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.size(0) == label.size(0):
                repeated_labels = label
            else:
                repeated_labels = label.repeat(self.num_scale)
            loss_val = -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)
            self.model.zero_grad()
            loss_val.backward(retain_graph=False)
        grad_map = x_.grad.detach().abs().mean(dim=1, keepdim=True)
        grad_map = (grad_map - grad_map.min()) / (grad_map.max() + 1e-8)
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        return grad_map

    def freq_perturbation(self, x, current_iter):
        if current_iter < 5:
            decay = 0.85
        elif current_iter < 15:
            decay = 0.93
        else:
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

    def region_based_cross_shuffle(self, x, saliency):
        B, C, H, W = x.shape
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
                if (random.random() < self.mix_ratio) or (mean_sal_i < mean_sal_j):
                    mixed_strips.append(strips_j[k])
                else:
                    mixed_strips.append(strips_i[k])
            x_mixed.append(torch.cat(mixed_strips, dim=2))
        return torch.cat(x_mixed, dim=0)

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

    def defense_emulate(self, x):
        if self.defense_type == 'jpeg':
            return self.jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def jpeg_compress(self, x, quality=75):
        return x

    def update_mix_ratio(self):
        if self.use_adaptive_mix:
            self.mix_ratio = max(0.01, self.mix_ratio * self.mix_decay)

    def shuffle(self, x):
        return super().shuffle(x)

    def transform(self, x, label=None, current_iter=0):
        _x = x
        if self.use_dim:
            _x = self.dim_transform(_x)
        if self.use_saliency_map and (label is not None):
            raise RuntimeError(
                "This transform function expects precomputed saliency for region-based mixing. "
                "Compute saliency outside the iteration to avoid multiple backward passes."
            )
        else:
            _x = self.cross_shuffle(_x)
        if self.use_defense_sim:
            _x = self.defense_emulate(_x)
        if self.use_freq_perturbation:
            _x = self.freq_perturbation(_x, current_iter=current_iter)
        shuffle_func = super(EnhancedCrossBSR, self).shuffle
        return torch.cat([shuffle_func(_x) for _ in range(self.num_scale)], dim=0)

    def saliency_region_loss(self, x_adv, x_orig, label):
        # 此处处理 x_adv 与 x_orig 的大小不匹配问题
        # x_adv 通常为重复后的对抗样本，其第一维为 B*num_scale，
        # 如果 x_orig 的第0维大小与 x_adv 不同，则需要将 x_orig 重复相应次数
        if x_adv.size(0) != x_orig.size(0):
            repeat_factor = x_adv.size(0) // x_orig.size(0)
            x_orig_rep = x_orig.repeat(repeat_factor, 1, 1, 1)
        else:
            x_orig_rep = x_orig
        diff = (x_adv - x_orig_rep).abs().mean(dim=1, keepdim=True)
        diff_sal = diff * self.compute_saliency_map(x_orig_rep, label)
        return diff_sal.mean()

    def get_loss(self, logits, label, x_adv=None, x_orig=None):
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

    def get_grad(self, loss, delta_or_x):
        grad = super(CrossBSR, self).get_grad(loss, delta_or_x)
        if self.use_sgm and len(self.gradients) > 0:
            shallow_grad = self.gradients[-1]
            self.gradients.clear()
            if shallow_grad.shape == grad.shape:
                grad += self.sgm_gamma * shallow_grad
        return grad

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        self.momentum = 0
        for i in tqdm(range(self.epoch), desc=f"Attack: {self.attack}"):
            if self.use_saliency_map:
                saliency_map = self.compute_saliency_map(data + delta, label)
                x_mixed = self.region_based_cross_shuffle(data + delta, saliency_map)
                if self.use_defense_sim:
                    x_mixed = self.defense_emulate(x_mixed)
                if self.use_freq_perturbation:
                    x_mixed = self.freq_perturbation(x_mixed, current_iter=i)
                shuffle_func = super(EnhancedCrossBSR, self).shuffle
                x_adv = torch.cat([shuffle_func(x_mixed) for _ in range(self.num_scale)], dim=0)
            else:
                x_adv = self.transform(data + delta, label=label, current_iter=i)
            logits = self.model(x_adv)
            loss = self.get_loss(logits, label, x_adv, data + delta)
            grad = self.get_grad(loss, delta)
            self.momentum = self.get_momentum(grad, self.momentum)
            delta = self.update_delta(delta, data, self.momentum, self.alpha)
            if self.saliency_save_dir is not None and self.use_saliency_map:
                sal_map = self.compute_saliency_map(data + delta, label)
                save_path = os.path.join(self.saliency_save_dir, f"saliency_iter_{i}.png")
                save_image(sal_map, save_path)
            self.update_mix_ratio()
        return delta.detach()

    def __del__(self):
        for handle in getattr(self, 'handles', []):
            handle.remove()