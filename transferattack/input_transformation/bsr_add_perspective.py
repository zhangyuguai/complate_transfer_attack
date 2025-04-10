#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本文件实现了多种对抗攻击方法：
1. BSR 攻击：通过块级洗牌和随机旋转扰动图像，
   增加模型对输入扰动的鲁棒性和攻击迁移性。
2. CrossBSR 攻击：在 BSR 基础上增加跨图像块混合，
   部分区域采用来自不同图像的块。
3. CrossBSR_Adaptive：在 CrossBSR 基础上引入动态混合比例，
   随着迭代逐步降低混合比例。
4. EnhancedCrossBSR 攻击：整合了多尺度变换（DIM）、浅层梯度增强（SGM）、
   防御模拟和额外的透视（Perspective）变换，用以提高攻击的迁移性。
   
使用示例：
    可通过命令行指定输入与输出目录以及攻击模型来运行本文件中的攻击逻辑.
    
注意：本文件中部分防御模拟函数（如 jpeg_compress）仅供示例，如需实际使用，
     可替换为对应的实现方案。
"""

import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

from ..utils import *
from ..gradient.mifgsm import MIFGSM

# ----------------------- BSR 攻击 -----------------------
class BSR(MIFGSM):
    """
    BSR 攻击
    “Boosting Adversarial Transferability by Block Shuffle and Rotation” 
    (https://arxiv.org/abs/2308.10299)
    
    每次迭代：
      1. 将输入图像按 spatial 维度分成若干 block。
      2. 随机洗牌 block 顺序。
      3. 对每个经过洗牌的 block 进行随机旋转（例如±24度）。
      4. 重复生成多个（num_scale）扰动版本，增加攻击多样性。
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

        # 块级随机旋转变换
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

    def get_length(self, length):
        """
        将长度为 length 的维度随机切分成 num_block 个不等段，切分结果之和等于 length。
        """
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += (length - rand_norm.sum())
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        """
        沿指定维度 dim 对块做洗牌。
        """
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        """
        对图像 x 进行随机旋转（±24度，双线性插值）。
        """
        return self.rotation_transform(x)

    def shuffle(self, x):
        """
        执行以下步骤：
          1. 随机选择两个 spatial 维度的顺序（如[2,3]或[3,2]）。
          2. 沿第一个选定的维度洗牌，将每个块进行随机旋转后，
             沿第二个维度再次洗牌。
          3. 将洗牌后的块按原来维度顺序拼接恢复图像。
        """
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        out_strips = []
        for x_strip in x_strips:
            # 在沿下一个维度洗牌前先随机旋转
            x_rotated = self.image_rotation(x_strip)
            out_strips.append(torch.cat(self.shuffle_single_dim(x_rotated, dims[1]), dim=dims[1]))
        return torch.cat(out_strips, dim=dims[0])

    def transform(self, x, **kwargs):
        """
        对输入 x 执行 BSR 变换：生成 num_scale 个经过 block 洗牌扰动的版本。
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)], dim=0)

    def get_loss(self, logits, label):
        """
        重复标签以匹配扰动版本数量，然后计算标准的交叉熵损失（或针对性攻击时取负值）。
        """
        repeated_labels = label.repeat(self.num_scale)
        if self.targeted:
            return -self.loss(logits, repeated_labels)
        else:
            return self.loss(logits, repeated_labels)


# ----------------------- CrossBSR 攻击 -----------------------
class CrossBSR(BSR):
    """
    CrossBSR 攻击：在 BSR 基础上增加跨图像块混合操作。
    参数 mix_ratio 控制选取其他图像块的概率。
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
        跨图像混合操作：
          对于批次中每个图像 x_i，随机选取另一图像 x_j，
          将它们沿高度分割成 num_block 个条带，
          根据 mix_ratio 以一定概率选择 x_j 的条带，否则选择 x_i 的对应条带，
          最后沿高度拼接恢复图像。
        """
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
        """
        先执行跨图像混合，再按标准 BSR 方式进行多尺度洗牌。
        """
        x_mixed = self.cross_shuffle(x)
        return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)


# ----------------------- CrossBSR_Adaptive 攻击 -----------------------
class CrossBSR_Adaptive(CrossBSR):
    """
    CrossBSR_Adaptive 攻击：在 CrossBSR 的基础上引入动态混合比例，
    每次迭代后根据 mix_decay 衰减混合比例。
    """
    def __init__(self, mix_init=0.5, mix_decay=0.95, **kwargs):
        super().__init__(mix_ratio=mix_init, **kwargs)
        self.mix_decay = mix_decay

    def update_mix_ratio(self):
        self.mix_ratio *= self.mix_decay

    def forward(self, data, label, **kwargs):
        for epoch in range(self.epoch):
            self.update_mix_ratio()
            delta = super().forward(data, label, **kwargs)
        return delta


# ----------------------- EnhancedCrossBSR 攻击 -----------------------
class EnhancedCrossBSR(CrossBSR):
    """
    EnhancedCrossBSR 攻击：集成多策略以提高攻击迁移性，包括：
      1. 动态混合比例（Adaptive Mix）。
      2. 多尺度变换（DIM）。
      3. 浅层梯度增强（SGM）。
      4. 防御模拟（如 JPEG 压缩模拟）。
      5. 附加的透视变换，利用 CNN 对局部几何与视角变化的敏感性。
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
            use_dim=True,                # 启用多尺度变换
            dim_resize_range=(0.8, 1.2),   # 缩放范围
            use_adaptive_mix=True,         # 启用动态混合
            mix_decay=0.95,                # 混合比例衰减率
            use_sgm=True,                  # 启用浅层梯度增强（SGM）
            sgm_gamma=0.5,                 # 浅层梯度权重
            use_defense_sim=False,         # 启用防御模拟
            defense_type='jpeg',           # 防御模拟类型
            perspective=True,              # 启用透视变换
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
        self.use_dim = use_dim
        self.dim_resize_range = dim_resize_range
        self.use_adaptive_mix = use_adaptive_mix
        self.mix_decay = mix_decay
        self.use_sgm = use_sgm
        self.sgm_gamma = sgm_gamma
        self.use_defense_sim = use_defense_sim
        self.defense_type = defense_type
        self.perspective = perspective

        if self.use_sgm:
            self._init_sgm_hooks()

    def _init_sgm_hooks(self):
        """注册浅层梯度钩子"""
        self.gradients = []
        self.handles = []
        for name, module in self.model.named_modules():
            if 'layer1' in name:  # 根据具体模型结构调整，此处以 ResNet 的 layer1 为例
                handle = module.register_backward_hook(self._save_grad)
                self.handles.append(handle)
                break

    def _save_grad(self, module, grad_input, grad_output):
        """保存浅层梯度"""
        self.gradients.append(grad_output[0].detach())

    def apply_random_perspective(self, x, distortion_scale=0.5, prob=0.5):
        """
        对输入 x 执行随机透视变换。
        Args:
            x (Tensor): 形状为 [B, C, H, W]。
            distortion_scale (float): 控制扭曲程度（0到1之间）。
            prob (float): 进行变换的概率。
        Returns:
            Tensor: 透视变换后的图像，形状与 x 相同。
        """
        perspective_transform = T.RandomPerspective(
            distortion_scale=distortion_scale,
            p=prob,
            interpolation=T.InterpolationMode.BILINEAR
        )
        return perspective_transform(x)

    def transform_with_perspective(self, x, distortion_scale=0.5, prob=0.5):
        """
        在标准转换前应用透视变换。
        """
        x_perspective = self.apply_random_perspective(x, distortion_scale, prob)
        return super(self.__class__, self).transform(x_perspective)

    def dim_transform(self, x):
        """
        多尺度变换（DIM）：随机缩放图像，然后通过正填充或负填充（裁剪）恢复到原尺寸。
        """
        resize_factor = random.uniform(*self.dim_resize_range)
        original_h, original_w = x.size(2), x.size(3)
        new_h = int(original_h * resize_factor)
        new_w = int(original_w * resize_factor)
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        def calculate_pad_crop(orig, new):
            delta = orig - new
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
        防御模拟：目前支持 'jpeg'（模拟 JPEG 压缩）和 'gaussian'（添加高斯噪声）。
        """
        if self.defense_type == 'jpeg':
            return jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def transform(self, x, **kwargs):
        """
        集成数据增强流程：
          1. 多尺度变换（DIM）；
          2. 透视变换（如果启用）；
          3. 跨图像块混合；
          4. 防御模拟（如果启用）；
          5. 标准 BSR 洗牌，并复制 num_scale 个版本。
        """
        if self.use_dim:
            x = self.dim_transform(x)
        if self.perspective:
            x = self.apply_random_perspective(x)
        x_mixed = self.cross_shuffle(x)
        if self.use_defense_sim:
            x_mixed = self.defense_emulate(x_mixed)
        return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)

    def get_grad(self, loss, x_trans):
        """
        计算梯度，并融合浅层梯度（SGM），以增强攻击对浅层特征的迁移性。
        """
        grad = super().get_grad(loss, x_trans)
        if self.use_sgm and len(self.gradients) > 0:
            batch_size = x_trans.size(0)
            shallow_grad = self.gradients[-1]
            shallow_grad = shallow_grad.view(batch_size, self.num_scale, *shallow_grad.shape[1:]).mean(dim=1)
            if shallow_grad.size(1) > 3:
                shallow_grad = shallow_grad.mean(dim=1, keepdim=True)
                shallow_grad = shallow_grad.repeat(1, 3, 1, 1)
            shallow_grad = F.interpolate(shallow_grad, size=grad.shape[-2:], mode='bilinear', align_corners=False)
            grad += self.sgm_gamma * shallow_grad
            self.gradients = []
        return grad

    def update_mix_ratio(self):
        """
        动态调整混合比例：每次迭代后 mix_ratio 根据 mix_decay 衰减，最小值为 0.01。
        """
        if self.use_adaptive_mix:
            self.mix_ratio = max(0.01, self.mix_ratio * self.mix_decay)

    def forward(self, data, label, **kwargs):
        """
        攻击过程：每轮迭代更新混合比例，并计算扰动 delta。
        """
        for _ in range(self.epoch):
            self.update_mix_ratio()
            delta = super().forward(data, label, **kwargs)
        return delta

    def __del__(self):
        """
        清理注册的钩子，防止内存泄漏。
        """
        for handle in getattr(self, 'handles', []):
            handle.remove()