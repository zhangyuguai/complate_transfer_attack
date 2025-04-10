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
      - 移除了在 Integrated Gradients 计算后 x.requires_grad_(False) 的操作，
        避免 RuntimeError: you can only change requires_grad flags of leaf variables。
      - 其余流程与先前的版本一致，仅供示例。
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
            # DIM
            use_dim=True,
            dim_resize_range=(0.8, 1.2),
            # Adaptive mix
            use_adaptive_mix=True,
            mix_decay=0.95,
            # SGM
            use_sgm=True,
            sgm_gamma=0.5,
            # Defense simulate
            use_defense_sim=False,
            defense_type='jpeg',
            # Saliency
            use_saliency_map=True,
            saliency_weight=0.1,
            gradcam_target_layer="layer1",
            # Freq domain
            use_freq_perturbation=False,
            freq_epsilon=0.05,
            # integrated grad
            use_integrated_grad=True,
            ig_steps=50,
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

        # 随机旋转
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

        if self.use_sgm:
            self._init_sgm_hooks()

        self.momentum = 0  # MIFGSM的momentum buffer

    def _init_sgm_hooks(self):
        """初始化SGM相关的backward hook。"""
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
        """随机切分维度长度为 num_block 段，并返回各段的大小。"""
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += (length - rand_norm.sum())
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        """对指定 dim 的 block 做洗牌。"""
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        """对 x 做一次随机旋转。"""
        return self.rotation_transform(x)

    # ------------------------- 显著性计算 -------------------------
    def grad_cam_saliency(self, x, label, target_layer="layer1"):
        """
        使用 Grad-CAM 计算显著性图:
          1) 注册前向/反向钩子
          2) 前向传播获取 activation
          3) 反向传播获取 gradients
          4) cam = ReLU(activation * GAP(gradients))
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
        积分梯度 (Integrated Gradients) 显著性图：
          baseline = 0
          IG = (x - baseline) * 累加(grad(F(x_i))) / steps
        """
        baseline = torch.zeros_like(x)
        x_diff = x - baseline

        # 不强制修改 x 的 requires_grad，此处 x 通常不是叶子节点
        ig = torch.zeros_like(x)

        for alpha in torch.linspace(0, 1, steps):
            x_interpolated = baseline + alpha * x_diff
            x_interpolated = x_interpolated.detach().requires_grad_(True)

            logits = self.model(x_interpolated)
            base_loss = -self.loss(logits, label) if self.targeted else self.loss(logits, label)
            self.model.zero_grad()
            base_loss.backward(retain_graph=False)

            if x_interpolated.grad is not None:
                ig += x_interpolated.grad

        # 计算 IG
        ig = x_diff * ig / steps
        # 取绝对值并对通道做平均
        ig_sal = ig.abs().mean(dim=1, keepdim=True)
        ig_sal = ig_sal - ig_sal.min()
        ig_sal = ig_sal / (ig_sal.max() + 1e-8)
        return ig_sal

    # ------------------------- 图像变换、混合 -------------------------
    def cross_shuffle(self, x):
        """Cross-image mixing: 每张图随机选另一张图混合 block。"""
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
        显著性区域引导的跨图像混合：
          使用 Grad-CAM 或 IG 计算显著性，再基于显著值决定如何选取 block。
        """
        B, C, H, W = x.shape
        if self.use_integrated_grad:
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
        """多尺度变换 (DIM)。"""
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
        BSR 里的 block shuffle + 随机旋转。
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

    def defense_emulate(self, x):
        """简单的防御模拟。"""
        if self.defense_type == 'jpeg':
            return self.jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def jpeg_compress(self, x, quality=75):
        # 仅作示例
        return x

    def freq_perturbation(self, x):
        """频域扰动."""
        Xf = torch.fft.fftn(x, dim=(-2, -1))
        noise_real = torch.randn_like(Xf.real)
        noise_imag = torch.randn_like(Xf.imag)

        Xf_real_perturbed = Xf.real + self.freq_epsilon * noise_real
        Xf_imag_perturbed = Xf.imag + self.freq_epsilon * noise_imag
        Xf_perturbed = torch.complex(Xf_real_perturbed, Xf_imag_perturbed)

        perturbed = torch.fft.ifftn(Xf_perturbed, dim=(-2, -1)).real
        return torch.clamp(perturbed, 0.0, 1.0)

    def transform(self, x, label=None, momentum=None, **kwargs):
        """
        核心 transform 流程：
          1. 多尺度变换 (DIM)
          2. 显著性区域混合 或 普通 cross_shuffle
          3. 防御模拟
          4. 频域扰动
          5. BSR 标准 shuffle 并 replicate num_scale 次
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

        return torch.cat([self.shuffle(_x) for _ in range(self.num_scale)], dim=0)

    # ------------------------- 损失函数 & 训练逻辑 -------------------------
    def saliency_region_loss(self, x_adv, x_orig, label):
        """
        显著性区域的额外损失约束，不使用 with torch.no_grad()，
        以免在 grad_cam_saliency 中构建图时被禁止梯度追踪。
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
        如果 is_for_saliency=True，则说明只是显著性计算，不做 num_scale repeat。
        否则，对抗攻击阶段要与 transform(...) 保持一致。
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
        """整合 SGM 的梯度计算。"""
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
            # 如果是定向攻击 label=[gt_label, target_label]
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
        """清理钩子。"""
        for handle in getattr(self, 'handles', []):
            handle.remove()