import torch
import pywt
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class BSR(MIFGSM):
    """
    BSR 攻击
    “Boosting Adversarial Transferability by Block Shuffle and Rotation”
    该类通过对图像分块、随机打乱以及随机旋转来生成对抗样本，提升攻击的多样性和迁移性。
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
        # 指定随机旋转范围为 ±24 度（使用双线性插值）
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

    def get_length(self, length):
        """
        将长度为 length 的维度随机切分成 num_block 个不等的部分，
        并保证各部分长度之和等于 length。
        """
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += (length - rand_norm.sum())
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        """
        沿指定维度 dim 分块后随机打乱。
        """
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        """
        对图像 x 执行随机旋转（±24度）。
        """
        return self.rotation_transform(x)

    def shuffle(self, x):
        """
        先在一个空间维度上打乱后，在另一空间维度上经过随机旋转后再打乱，
        并将结果拼接起来。
        """
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
        """
        标准 BSR 变换：生成 num_scale 个经过 block shuffle 处理的版本。
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)], dim=0)

    def get_loss(self, logits, label):
        """
        重复标签以匹配扩展后的批次并计算交叉熵损失。
        """
        repeated_labels = label.repeat(self.num_scale)
        return -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)


class CrossBSR(BSR):
    """
    CrossBSR 攻击：
      在 BSR 基础上增加跨图像块混合，即每张图有一定概率被替换部分块为其他随机图像的块。
      参数 mix_ratio 控制替换概率。
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
        对每张图，从批次中随机挑选另一张图，
        按高度分块后以一定概率对该块进行替换生成混合图像。
        """
        B, C, H, W = x.shape
        x_mixed = []
        split_h = self.get_length(H)
        for i in range(B):
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i:i+1]
            x_j = x[j:j+1]
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
        """
        先进行跨图像混合，再应用标准 BSR 变换复制 num_scale 个版本。
        """
        x_mixed = self.cross_shuffle(x)
        return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)


class EnhancedCrossBSR(CrossBSR):
    """
    增强版 CrossBSR 攻击，融合多个策略：
      1. 动态混合比例（Adaptive Mix）
      2. 多尺度输入变换（DIM）
      3. 浅层梯度增强（SGM）
      4. 防御模拟（NRDM）
      5. 基于 Grad-CAM 的显著性区域引导混合
      6. 频域扰动（FFT-domain perturbation）
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
            gradcam_target_layer="layer1",
            use_freq_perturbation=True,
            freq_epsilon=0.05,
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

        if self.use_sgm:
            self._init_sgm_hooks()

        self.momentum = 0

    # 1. 浅层梯度增强（SGM）
    def _init_sgm_hooks(self):
        self.gradients = []
        self.handles = []
        for name, module in self.model.named_modules():
            if 'layer1' in name:
                handle = module.register_backward_hook(self._save_grad)
                self.handles.append(handle)
                break

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    # 2. Grad-CAM 显著性图计算
    def grad_cam_saliency(self, x, label, target_layer="layer1"):
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

    # 3. 频域扰动
    def freq_perturbation(self, x):
        Xf = torch.fft.fftn(x, dim=(-2, -1))
        noise_real = torch.randn_like(Xf.real)
        noise_imag = torch.randn_like(Xf.imag)
        Xf_real_perturbed = Xf.real + self.freq_epsilon * noise_real
        Xf_imag_perturbed = Xf.imag + self.freq_epsilon * noise_imag
        Xf_perturbed = torch.complex(Xf_real_perturbed, Xf_imag_perturbed)
        perturbed = torch.fft.ifftn(Xf_perturbed, dim=(-2, -1)).real
        return torch.clamp(perturbed, 0.0, 1.0)

    # 4. 基于显著性区域引导的跨图像混合
    def region_based_cross_shuffle(self, x, label):
        B, C, H, W = x.shape
        x_mixed = []
        saliency = self.grad_cam_saliency(x, label, target_layer=self.gradcam_target_layer)
        saliency = saliency.clamp(min=0., max=1.)
        split_h = self.get_length(H)
        for i in range(B):
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i:i+1]
            x_j = x[j:j+1]
            s_i = saliency[i:i+1]
            s_j = saliency[j:j+1]
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

    # 5. 多尺度变换 (DIM)
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

    # 6. 防御模拟
    def defense_emulate(self, x):
        if self.defense_type == 'jpeg':
            return self.jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def jpeg_compress(self, x, quality=75):
        # 仅为示例，不实现具体 JPEG 算法
        return x

    # 7. 全流程变换
    def update_mix_ratio(self):
        if self.use_adaptive_mix:
            self.mix_ratio = max(0.01, self.mix_ratio * self.mix_decay)

    def transform(self, x, label=None, saliency_map=None, momentum=None, **kwargs):
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
        return torch.cat([self.shuffle(_x) for _ in range(self.num_scale)], dim=0)

    # 8. 显著性区域损失
    def saliency_region_loss(self, x_adv, x_orig, label):
        with torch.no_grad():
            cam_orig = self.grad_cam_saliency(x_orig, label, target_layer=self.gradcam_target_layer)
        diff = (x_adv - x_orig).abs().mean(dim=1, keepdim=True)
        diff_sal = diff * cam_orig
        return diff_sal.mean()

    # 9. 综合损失
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

    # 10. 梯度计算（整合 SGM）
    def get_grad(self, loss, delta_or_x):
        """
        修改后的 get_grad：允许未使用的 Tensor（allow_unused=True）以避免 RuntimeError。
        """
        # 在此调用 autograd.grad 时，传入 allow_unused=True
        grad_list = torch.autograd.grad(loss, delta_or_x, retain_graph=False, create_graph=False, allow_unused=True)
        # 如果未计算出梯度，则返回零梯度
        if grad_list[0] is None:
            grad = torch.zeros_like(delta_or_x)
        else:
            grad = grad_list[0]
        if self.use_sgm and len(self.gradients) > 0:
            shallow_grad = self.gradients[-1]
            batch_size = delta_or_x.size(0)
            shallow_grad = shallow_grad.view(batch_size, self.num_scale, *shallow_grad.shape[1:]).mean(dim=1)
            if shallow_grad.size(1) > 3:
                shallow_grad = shallow_grad.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            shallow_grad = F.interpolate(shallow_grad, size=grad.shape[-2:], mode='bilinear', align_corners=False)
            grad += self.sgm_gamma * shallow_grad
            self.gradients = []
        return grad

    # 11. 统一 forward
    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        self.momentum = 0
        for _ in tqdm(range(self.epoch), desc=f"Attack: {self.attack} attack_model: {self.model_name}"):
            logits = self.model(self.transform(data + delta, momentum=self.momentum))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            self.momentum = self.get_momentum(grad, self.momentum)
            delta = self.update_delta(delta, data, self.momentum, self.alpha)
        return delta.detach()

    def __del__(self):
        for handle in getattr(self, 'handles', []):
            handle.remove()


class WaveletEnhancedBSR(EnhancedCrossBSR):
    """
    在 EnhancedCrossBSR 的基础上加入小波扰动模块，该模块在 transform 流程的末尾
    使用离散小波变换（DWT）对图像在 detail 分量上添加随机扰动，
    再重建图像。该过程使用 no_grad 避免破坏反向传播。
    """
    def __init__(
            self,
            wavelet='haar',
            wavelet_level=1,
            wavelet_coef_scale=0.02,
            **kwargs
    ):
        """
        参数:
          wavelet: 小波类型（如 'haar', 'db1', 'coif1' 等）
          wavelet_level: 小波分解层数
          wavelet_coef_scale: 对 detail 系数扰动的比例
        """
        super().__init__(**kwargs)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_coef_scale = wavelet_coef_scale

    def wavelet_perturbation(self, x):
        """
        对输入 x 在小波域中添加小扰动，先进行 DWT，扰动 detail 系数，
        然后重建图像。此过程不计入梯度计算，因此需要用 no_grad 包裹。
        """
        b, c, h, w = x.shape
        x_perturbed = []
        for i in range(b):
            channels_perturbed = []
            for ch in range(c):
                single_channel = x[i, ch, :, :].detach().cpu().numpy()
                coeffs = pywt.wavedec2(single_channel, self.wavelet, level=self.wavelet_level)
                new_coeffs = [coeffs[0]]
                for detail_level in coeffs[1:]:
                    cH, cV, cD = detail_level
                    # 为 detail 系数添加随机噪声
                    cH += self.wavelet_coef_scale * (2 * torch.rand_like(torch.tensor(cH)) - 1).numpy()
                    cV += self.wavelet_coef_scale * (2 * torch.rand_like(torch.tensor(cV)) - 1).numpy()
                    cD += self.wavelet_coef_scale * (2 * torch.rand_like(torch.tensor(cD)) - 1).numpy()
                    new_coeffs.append((cH, cV, cD))
                perturbed_ch = pywt.waverec2(new_coeffs, self.wavelet)
                perturbed_ch = torch.tensor(perturbed_ch, device=x.device).clamp(0.0, 1.0)
                if perturbed_ch.shape != (h, w):
                    perturbed_ch = F.interpolate(
                        perturbed_ch.view(1, 1, *perturbed_ch.shape),
                        size=(h, w), mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                channels_perturbed.append(perturbed_ch)
            channels_perturbed = torch.stack(channels_perturbed, dim=0)
            x_perturbed.append(channels_perturbed)
        return torch.stack(x_perturbed, dim=0)

    def transform(self, x, label=None, saliency_map=None, momentum=None, **kwargs):
        """
        重写 transform：先使用 EnhancedCrossBSR 的 transform，再对输出图像在小波域加扰，
        此操作以 no_grad 模式执行以避免破坏梯度计算。
        """
        _x = super().transform(x, label, saliency_map, momentum, **kwargs)
        with torch.no_grad():
            _x = self.wavelet_perturbation(_x)
        return _x