import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class BSR(MIFGSM):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'(https://arxiv.org/abs/2308.10299)

    In each iteration:
      1. The input is subdivided into blocks (num_block x num_block).
      2. The blocks are shuffled in a random order along spatial dimensions.
      3. Each shuffled block may also be randomly rotated (e.g. within ±24 degrees).
      4. Multiple copies (num_scale) of these shuffled images are fed to the model,
         increasing the attack's diversity.

    Arguments:
        model_name (str): The name of the surrogate model for the attack.
        epsilon (float): The perturbation budget.
        alpha (float): The step size in each iteration (if norm='linfty').
        epoch (int): Number of attack iterations.
        decay (float): Decay factor for momentum calculation.
        num_scale (int): Number of shuffled copies used per iteration.
        num_block (int): The number of blocks along each dimension.
        targeted (bool): Targeted or untargeted attack.
        random_start (bool): Whether to use random initialization for delta.
        norm (str): Norm type of perturbation ('l2' or 'linfty').
        loss (str): Loss function to use (e.g., 'crossentropy').
        device (torch.device): The torch device to run the attack on.
        attack (str): Attack name for logging or reference.

    Example usage:
        python main.py --input_dir ./path/to/data --output_dir ./adv_data/bsr/resnet18 --attack bsr --model resnet18
        python main.py --input_dir ./path/to/data --output_dir ./adv_data/bsr/resnet18 --eval
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
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block

        # For block rotation
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

    def get_length(self, length):
        """
        Randomly split a dimension (length) into num_block segments of uneven sizes.
        Returns a tuple with each segment length that sums to 'length'.
        """
        # E.g. sample 'num_block' random values from uniform(2)
        rand = np.random.uniform(2, size=self.num_block)
        # Normalize so their sum = length
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        # Fix rounding in case sums differ from original
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
        Random rotation of x within ±24 degrees (bilinear).
        """
        return self.rotation_transform(x)

    def shuffle(self, x):
        """
        1. Randomly choose an order of dimensions (2, 3) or (3, 2).
        2. Shuffle blocks in the first dimension from dims,
           then apply random rotation, and shuffle blocks in the second dimension.
        3. Reconstruct the blocks.
        """
        dims = [2, 3]
        random.shuffle(dims)

        x_strips = self.shuffle_single_dim(x, dims[0])
        out_strips = []
        for x_strip in x_strips:
            # Apply random rotation before the next shuffle
            x_rotated = self.image_rotation(x_strip)
            out_strips.append(
                torch.cat(self.shuffle_single_dim(x_rotated, dims[1]), dim=dims[1])
            )
        return torch.cat(out_strips, dim=dims[0])

    def transform(self, x, **kwargs):
        """
        BSR transform step: replicate the shuffled images num_scale times.
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)], dim=0)

    def get_loss(self, logits, label):
        """
        Repeat the label to match num_scale, then compute the standard or reversed loss.
        """
        repeated_labels = label.repeat(self.num_scale)
        return -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)


class CrossBSR(BSR):
    """
    A variant of BSR that introduces cross-image block mixing before the main shuffle.
    mix_ratio controls the probability of picking blocks from another (random) image
    in the batch. The rest of the BSR pipeline remains the same.
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
            attack=attack
        )
        self.mix_ratio = mix_ratio

    def cross_shuffle(self, x):
        """
        Cross-image shuffle logic:
          1. For each image x_i in the batch, pick a random x_j.
          2. Split x_i and x_j into horizontal strips (num_block in the H dimension).
          3. Rebuild the image by selectively picking strips from x_i or x_j,
             controlled by mix_ratio (0.0 ~ 1.0).

        The result is a batch of images where each image is part original, part from another random image.
        """
        B, C, H, W = x.shape
        x_mixed = []

        # Precompute splits along H dimension
        split_h = self.get_length(H)

        for i in range(B):
            # Randomly pick another image index j
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i : i+1]  # shape [1, C, H, W]
            x_j = x[j : j+1]

            # Split into strips along height
            strips_i = x_i.split(split_h, dim=2)  # each shape ~ [1, C, h_i, W]
            strips_j = x_j.split(split_h, dim=2)

            # Build the mixed image from strips
            mixed_strips = []
            for k in range(self.num_block):
                if random.random() < self.mix_ratio:
                    mixed_strips.append(strips_j[k])
                else:
                    mixed_strips.append(strips_i[k])
            x_mixed.append(torch.cat(mixed_strips, dim=2))

        return torch.cat(x_mixed, dim=0)  # shape [B, C, H, W]

    def transform(self, x, **kwargs):
        """
        1. Mix blocks across images in the batch (cross_shuffle).
        2. Perform regular BSR transform with multiple scales.
        """
        #print(f"-------------------- CrossBSR in action (mix_ratio={self.mix_ratio}) --------------------")
        x_mixed = self.cross_shuffle(x)
        # Now apply standard BSR shuffle steps; replicate num_scale times
        return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)

# class CrossBSR_DIM(CrossBSR):
#     def __init__(self, resize_rate=0.85, **kwargs):
#         super().__init__(**kwargs)
#         self.resize_rate = resize_rate  # 缩放比例（如0.8~1.2）

#     def dim_transform(self, x):
#         # 随机缩放CrossBSR_DIM
#         new_size = int(x.size(2) * self.resize_rate)
#         x_resized = F.interpolate(x, size=new_size, mode='bilinear')
#         # 随机填充至原尺寸
#         pad_left = random.randint(0, x.size(2)-new_size)
#         pad_top = random.randint(0, x.size(3)-new_size)
#         return F.pad(x_resized, (pad_left, x.size(3)-new_size-pad_left,
#                              pad_top, x.size(2)-new_size-pad_top))

#     def transform(self, x, **kwargs):
#         x_scaled = self.dim_transform(x)  # 先尺度变换
#         x_mixed = self.cross_shuffle(x_scaled)  # 再跨图像分块
#         return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)


class EnhancedCrossBSR(CrossBSR):
    """
    增强版CrossBSR攻击，整合多策略：
    1. 动态混合比例（Adaptive Mix）
    2. 多尺度输入变换（DIM）
    3. 浅层梯度增强（SGM）
    4. 防御模拟（NRDM）
    现示例添加了：显著性区域引导的跨图像混合 + 显著性区域损失 (saliency region loss)
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
            # 跨图像混合
            mix_ratio=0.05,
            # 扩展模块参数
            use_dim=True,                # 启用多尺度变换
            dim_resize_range=(0.8, 1.2), # 缩放范围
            use_adaptive_mix=True,       # 启用动态混合
            mix_decay=0.95,              # 混合比例衰减率
            use_sgm=True,                # 启用浅层梯度增强
            sgm_gamma=0.5,               # 浅层梯度权重
            use_defense_sim=False,       # 启用防御模拟
            defense_type='jpeg',         # 防御类型
            # 新增：显著性区域处理参数
            use_saliency_map=True,
            saliency_weight=0.1,         # 显著性区域损失在总损失中的权重
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

        if self.use_sgm:
            self._init_sgm_hooks()

    # -----------------------------
    # 1. 浅层梯度增强
    # -----------------------------
    def _init_sgm_hooks(self):
        self.gradients = []
        self.handles = []
        # 注册浅层梯度钩子 (示例：ResNet的layer1)
        for name, module in self.model.named_modules():
            if 'layer1' in name:
                handle = module.register_backward_hook(self._save_grad)
                self.handles.append(handle)
                break

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def dim_transform(self, x):
        resize_factor = random.uniform(*self.dim_resize_range)
        original_h, original_w = x.size(2), x.size(3)
        new_h = int(original_h * resize_factor)
        new_w = int(original_w * resize_factor)
        x_resized = F.interpolate(
            x, size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )

        def calculate_pad_crop(original, new):
            delta = original - new
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
        """
        防御模拟：示例仅演示 JPEG 或 Gaussian 噪声
        """
        if self.defense_type == 'jpeg':
            return self.jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def jpeg_compress(self, x, quality=75):
        """
        简化版JPEG压缩模拟，仅示例化。
        真实实现需要 PIL 或类似库进行编码解码。
        """
        # 这里暂时返回原图
        return x

    # -----------------------------
    # 2. 显著性图计算 & 动态混合区域
    # -----------------------------
    def compute_saliency_map(self, x, label):
        """
        示例：基于简单的梯度绝对值来模拟显著性图 (saliency map)。
        可替换为更真实的Grad-CAM等方法。
        返回 shape [B, 1, H, W] 的显著性。
        """
        x = x.clone().detach().requires_grad_(True)
        logits = self.model(x)
        loss = self.get_loss(logits, label)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        saliency = grad.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        return saliency.detach()

    def region_based_cross_shuffle(self, x, label):
        """
        在 cross_shuffle 基础上，使用显著性图来动态决定哪些区域从其它图中混合。
        在显著区域出现时，更倾向于交换，以加强扰动影响。
        """
        B, C, H, W = x.shape
        x_mixed = []

        # 获取显著性图
        saliency = self.compute_saliency_map(x, label)  # [B,1,H,W]
        saliency = saliency / (saliency.max() + 1e-8)    # 归一化

        # 先计算分块大小
        split_h = self.get_length(H)  # [h1,h2,...,hK] sum=H

        for i in range(B):
            j = torch.randint(low=0, high=B, size=(1,)).item()
            x_i = x[i : i+1]      # shape [1,C,H,W]
            x_j = x[j : j+1]
            s_i = saliency[i : i+1]  # shape [1,1,H,W]
            s_j = saliency[j : j+1]

            strips_i = x_i.split(split_h, dim=2)
            strips_j = x_j.split(split_h, dim=2)

            # 显著性拆分
            s_strips_i = s_i.split(split_h, dim=2)
            s_strips_j = s_j.split(split_h, dim=2)

            mixed_strips = []
            for k in range(self.num_block):
                # 计算当前条带 s_i, s_j 的平均显著性
                mean_sal_i = s_strips_i[k].mean().item()
                mean_sal_j = s_strips_j[k].mean().item()

                # 倾向于交换显著性更高的区域
                # 我们这里示例：如果 i 的显著性更高，则用 j 的块替换，以增大对抗性
                # 也可反之或其它逻辑
                if (random.random() < self.mix_ratio) or (mean_sal_i > mean_sal_j):
                    mixed_strips.append(strips_j[k])
                else:
                    mixed_strips.append(strips_i[k])

            x_mixed.append(torch.cat(mixed_strips, dim=2))

        return torch.cat(x_mixed, dim=0)

    # -----------------------------
    # 3. 新 transform (可选择显著性区域混合)
    # -----------------------------
    def transform(self, x, label=None, **kwargs):
        # 多尺度变换
        if self.use_dim:
            x = self.dim_transform(x)

        # 是否先进行显著性区域引导的跨图像混合
        if self.use_saliency_map and label is not None:
            x_mixed = self.region_based_cross_shuffle(x, label)
        else:
            # 若未启用显著性图或无标签，可回退到原始 cross_shuffle
            x_mixed = self.cross_shuffle(x)

        # 防御模拟
        if self.use_defense_sim:
            x_mixed = self.defense_emulate(x_mixed)

        # BSR标准变换
        return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)

    # -----------------------------
    # 4. 显著性区域损失 (saliency region loss)
    # -----------------------------
    def saliency_region_loss(self, x_adv, x_orig, label):
        """
        示例：对显著区域差异施加约束，如在最显著处增加扰动能带来更大对抗效果。
        具体形式可根据攻击目标而定。
        这里简单演示：在显著区域对 (x_adv - x_orig) 的范数做加权。
        """
        # 计算原图显著性
        with torch.no_grad():
            s_map = self.compute_saliency_map(x_orig, label)  # [B,1,H,W]
            s_map = s_map / (s_map.max() + 1e-8)

        diff = (x_adv - x_orig).abs()  # [B,C,H,W]
        # 将 diff 与 s_map 相乘
        diff_sal = diff.mean(dim=1, keepdim=True) * s_map  # [B,1,H,W], 用均值简化通道
        # 取均值作为损失
        return diff_sal.mean()

    # -----------------------------
    # 5. 整合损失
    # -----------------------------
    def get_loss(self, logits, label, x_adv=None, x_orig=None):
        """
        在原有的 Cross-Entropy 基础上，加入显著性区域损失。
        """
        if isinstance(label, int):
            label = torch.tensor([label], dtype=torch.long)
        repeated_labels = label.repeat(self.num_scale)
        base_loss = -self.loss(logits, repeated_labels) if self.targeted else self.loss(logits, repeated_labels)

        if self.use_saliency_map and (x_adv is not None) and (x_orig is not None):
            # 只重复 label 以匹配 batch
            # x_adv 已经是 [B*num_scale, ...], 可能需要分块还原
            # 考虑最简单情况：x_adv 与 x_orig 的 shape 一致
            sal_loss = self.saliency_region_loss(x_adv, x_orig, label)
            total_loss = base_loss + self.saliency_weight * sal_loss
        else:
            total_loss = base_loss

        return total_loss

    # -----------------------------
    # 6. 迭代过程
    # -----------------------------
    def get_grad(self, loss, x_trans):
        """梯度计算增强"""
        grad = super().get_grad(loss, x_trans)
        # 浅层梯度增强
        if self.use_sgm and len(self.gradients) > 0:
            shallow_grad = self.gradients[-1]  # [B*num_scale, C, H, W]

            # 新增维度平均逻辑
            batch_size = x_trans.size(0)   # 原始batch size
            shallow_grad = shallow_grad.view(batch_size, self.num_scale, *shallow_grad.shape[1:]).mean(dim=1)

            # 通道压缩
            if shallow_grad.size(1) > 3:
                shallow_grad = shallow_grad.mean(dim=1, keepdim=True)
                shallow_grad = shallow_grad.repeat(1, 3, 1, 1)

            # 上采样
            shallow_grad = F.interpolate(
                shallow_grad,
                size=grad.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            grad += self.sgm_gamma * shallow_grad
            self.gradients = []
        return grad

        return grad
    def update_mix_ratio(self):
        if self.use_adaptive_mix:
            self.mix_ratio = max(0.01, self.mix_ratio * self.mix_decay)

    def fixed_forward(self, data, label, **kwargs):
        """
        修复设备冲突的示例forward方法，确保data和delta在同一设备上进行运算。
        """
        # 首先将 data 移到 self.device
        data = data.to(self.device)
        label = label.to(self.device)

        print(f"当前设备：{self.device}")

        # 确保 init_delta 返回的张量也在同一个device上
        delta = self.init_delta(data).to(self.device)

        # 使 adv_data 也在同一个device上
        adv_data = data + delta

        for i in range(self.epoch):
            self.update_mix_ratio()

            # transform 中也需要注意 x, label 在相同 device 上
            # transform 里若需要 label，也应确保 label.to(self.device)
            x_trans = self.transform(adv_data, label=label)
            x_trans = x_trans.to(self.device)
            x_trans.requires_grad_(True)

            # 计算 logits
            logits = self.model(x_trans.to(self.device))

            # 构造显著性区域损失
            loss = self.get_loss(logits, label, x_trans, adv_data)

            # 计算梯度
            grad = self.get_grad(loss, x_trans)
            grad_orig = grad.mean(dim=0, keepdim=True)

            # 使用 L∞ 更新方式示例
            if self.norm == 'linfty':
                update = grad_orig.sign()
                adv_data = adv_data + self.alpha * update
                delta = adv_data - data
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adv_data = torch.clamp(data + delta, 0.0, 1.0)

        return adv_data

    def __del__(self):
        for handle in getattr(self, 'handles', []):
            handle.remove()