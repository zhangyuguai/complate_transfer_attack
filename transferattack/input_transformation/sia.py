# import torch
# import torch.nn.functional as F
# from matplotlib import pyplot as plt
#
# from ..utils import *
# from ..gradient.mifgsm import MIFGSM
#
# import scipy.stats as st
# import numpy as np
#
# class SIA(MIFGSM):
#     """
#     SIA(Structure Invariant Attack)
#     'Structure Invariant Transformation for better Adversarial Transferability' (https://arxiv.org/abs/2309.14700)
#
#     This version draws visible boundary lines after each block transformation.
#     """
#
#     def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,
#                  num_scale=20, num_block=3, targeted=False, random_start=False,
#                  norm='linfty', loss='crossentropy', device=None, attack='SIA', **kwargs):
#         super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
#         self.num_scale = num_scale
#         self.num_block = num_block
#         self.kernel = self.gkern()
#         self.op = [
#             self.vertical_shift,
#             self.horizontal_shift,
#             self.vertical_flip,
#             self.horizontal_flip,
#             self.rotate180,
#             self.scale,
#             self.add_noise
#         ]
#
#     def vertical_shift(self, x):
#         _, _, w, _ = x.shape
#         step = np.random.randint(low=0, high=w, dtype=np.int32)
#         return x.roll(step, dims=2)
#
#     def horizontal_shift(self, x):
#         _, _, _, h = x.shape
#         step = np.random.randint(low=0, high=h, dtype=np.int32)
#         return x.roll(step, dims=3)
#
#     def vertical_flip(self, x):
#         return x.flip(dims=(2,))
#
#     def horizontal_flip(self, x):
#         return x.flip(dims=(3,))
#
#     def rotate180(self, x):
#         return x.rot90(k=2, dims=(2, 3))
#
#     def scale(self, x):
#         return torch.rand(1)[0] * x
#
#     def add_noise(self, x):
#         return torch.clamp(x + torch.zeros_like(x).uniform_(-16/255, 16/255), 0, 1)
#
#     def gkern(self, kernel_size=3, nsig=3):
#         # Create a 2D gaussian kernel for blurring
#         x = np.linspace(-nsig, nsig, kernel_size)
#         kern1d = st.norm.pdf(x)
#         kernel_raw = np.outer(kern1d, kern1d)
#         kernel = kernel_raw / kernel_raw.sum()
#         stack_kernel = np.stack([kernel, kernel, kernel])  # three channels
#         stack_kernel = np.expand_dims(stack_kernel, 1)     # shape (3,1,kernel_size,kernel_size)
#         return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)
#
#     def blur(self, x):
#         return F.conv2d(x, self.kernel, stride=1, padding='same', groups=3)
#
#     def blocktransform(self, x, choice=-1):
#         """
#         Randomly partition the image into num_block x num_block blocks,
#         perform random transformations in each block, then draw boundary lines.
#         """
#         _, _, w, h = x.shape
#         # Randomly pick boundaries for blocks
#         y_axis = [0] + np.random.choice(
#             list(range(1, h)), self.num_block - 1, replace=False
#         ).tolist() + [h]
#         x_axis = [0] + np.random.choice(
#             list(range(1, w)), self.num_block - 1, replace=False
#         ).tolist() + [w]
#
#         # Sort them so they form increasing boundary indices
#         y_axis.sort()
#         x_axis.sort()
#
#         x_copy = x.clone()
#         # Apply random ops to each block
#         for i, idx_x in enumerate(x_axis[1:]):
#             for j, idx_y in enumerate(y_axis[1:]):
#                 chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
#                 x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](
#                     x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y]
#                 )
#
#         # Draw boundary lines (here using black lines = 0 for visibility)
#         # Vertical boundaries
#         for vertical in x_axis:
#             x_copy[:, :, vertical:vertical+1, :] = 0
#         # Horizontal boundaries
#         for horizontal in y_axis:
#             x_copy[:, :, :, horizontal:horizontal+1] = 0
#
#         return x_copy
#
#     def transform(self, x, **kwargs):
#         """
#         Scale the input for BSR by creating num_scale shuffled copies,
#         each with random block transformations.
#         """
#         return torch.cat([self.blocktransform(x) for _ in range(self.num_scale)], dim=0)
#
#     def get_loss(self, logits, label):
#         """
#         Calculate the loss
#         """
#         return (
#             -self.loss(logits, label.repeat(self.num_scale))
#             if self.targeted else
#             self.loss(logits, label.repeat(self.num_scale))
#         )




import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import scipy.stats as st
import numpy as np

# 假设下列内容在本地项目中可用，若不在同层级或需其他引用路径，请自行调整
from ..utils import *
from ..gradient.mifgsm import MIFGSM

class SIA(MIFGSM):
    """
    SIA(Structure Invariant Attack)
    'Structure Invariant Transformation for better Adversarial Transferability' (https://arxiv.org/abs/2309.14700)

    该版本中，每个 Block 的转换后会额外绘制可见的 Block 边界。
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,
                 num_scale=20, num_block=3, targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='SIA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block

        # 高斯核，用于blur函数
        self.kernel = self.gkern()

        # 一组可选的结构不变变换
        self.op = [
            self.vertical_shift,
            self.horizontal_shift,
            self.vertical_flip,
            self.horizontal_flip,
            self.rotate180,
            self.scale,
            self.add_noise
        ]

    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x):
        return torch.rand(1)[0] * x

    def add_noise(self, x):
        return torch.clamp(x + torch.zeros_like(x).uniform_(-16/255, 16/255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        """
        创建一个2D高斯卷积核，用于blur函数
        """
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])  # 对每个通道应用相同核
        stack_kernel = np.expand_dims(stack_kernel, 1)     # (3,1,kernel_size,kernel_size)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def blur(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding='same', groups=3)

    def blocktransform(self, x, choice=-1):
        """
        随机将图像分块（num_block x num_block），对每个块执行随机选择的结构不变转换，并在每个块的边缘画出可见边界（黑线）。
        """
        _, _, w, h = x.shape
        # 采样纵横边界
        y_axis = [0] + np.random.choice(
            list(range(1, h)), self.num_block - 1, replace=False
        ).tolist() + [h]
        x_axis = [0] + np.random.choice(
            list(range(1, w)), self.num_block - 1, replace=False
        ).tolist() + [w]

        y_axis.sort()
        x_axis.sort()

        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](
                    x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y]
                )

        # 画出分块边缘（黑线）
        for vertical in x_axis:
            x_copy[:, :, vertical:vertical+1, :] = 0
        for horizontal in y_axis:
            x_copy[:, :, :, horizontal:horizontal+1] = 0

        return x_copy

    def transform(self, x, **kwargs):
        """
        将输入复制 num_scale 份，每份随机执行一次 blocktransform 变换后拼接。
        """
        return torch.cat([self.blocktransform(x) for _ in range(self.num_scale)], dim=0)

    def get_loss(self, logits, label):
        """
        计算损失
        """
        return (
            -self.loss(logits, label.repeat(self.num_scale))
            if self.targeted else
            self.loss(logits, label.repeat(self.num_scale))
        )

class SIAWithSaliency(SIA):
    """
    在原 SIA 攻击的基础上，增添结合显著性图的逻辑：
    1. transform_with_saliency: 根据显著性图决定对高显著区域做更强或更多变换。
    2. 用户可以自行在外部生成显著性图（如Grad-CAM、GB等），并在调用 forward/attack 时传入。
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1.,
                 num_scale=20, num_block=3, targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='SIA-Saliency', saliency_threshold=0.5, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, num_scale, num_block, targeted,
                         random_start, norm, loss, device, attack, **kwargs)
        self.saliency_threshold = saliency_threshold

    def blocktransform_with_saliency(self, x, saliency=None, choice=-1):
        """
        根据给定的 saliency（形状与图像匹配），在显著性高的Block中执行更强或更多变换。
        """
        _, _, w, h = x.shape
        # 分块
        y_axis = [0] + np.random.choice(
            list(range(1, h)), self.num_block - 1, replace=False
        ).tolist() + [h]
        x_axis = [0] + np.random.choice(
            list(range(1, w)), self.num_block - 1, replace=False
        ).tolist() + [w]

        y_axis.sort()
        x_axis.sort()

        x_copy = x.clone()
        # 若无 saliency 则退化到普通 blocktransform
        if saliency is None:
            return super().blocktransform(x_copy, choice)

        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                # 计算该Block区域的平均显著值
                block_saliency = saliency[:, x_axis[i]:idx_x, y_axis[j]:idx_y]
                mean_val = block_saliency.mean().item()

                # 若显著度超过阈值，则更高概率或更强变换
                # 这里示例：若 mean_val > saliency_threshold，额外随机两次变换叠加
                if mean_val > self.saliency_threshold:
                    chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                    # 连续两次变换
                    temp = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])
                    chosen2 = np.random.randint(0, high=len(self.op), dtype=np.int32)
                    transformed_block = self.op[chosen2](temp)
                else:
                    chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                    transformed_block = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = transformed_block

        # 画分块边
        for vertical in x_axis:
            x_copy[:, :, vertical:vertical+1, :] = 0
        for horizontal in y_axis:
            x_copy[:, :, :, horizontal:horizontal+1] = 0

        return x_copy

    def transform_with_saliency(self, x, saliency_list=None):
        """
        与原 transform 类似，只是每个复制都会结合相应的 saliency。
        saliency_list: 可能是 [B, H, W] 的列表；若 None, 则退化到基础 transform。
        """
        # 如果没有提供显著性图，则继续使用默认 transform
        if saliency_list is None:
            return super().transform(x)

        # x.shape = [B, C, H, W]
        # saliency_list 应与 batch 维度对应，这里简单示例：假设 B=1 或同大小即可
        # 这里将 x broadcast 到 num_scale 份，每一次调用 blocktransform_with_saliency
        outputs = []
        base_batch = x.shape[0]
        for _ in range(self.num_scale):
            # 针对每个 batch 样本单独进行变换
            x_copy = []
            for b in range(base_batch):
                # 选一个随机变化
                x_b = x[b:b+1]  # [1, C, H, W]
                sal_b = None if saliency_list is None else saliency_list[b]
                # 扩展axes使得 [H, W] -> [1, H, W] 方便与 x_b 对应
                if sal_b is not None and len(sal_b.shape) == 2:
                    sal_b = sal_b.unsqueeze(0)
                # 调用专用的 blocktransform_with_saliency
                x_transformed = self.blocktransform_with_saliency(x_b, sal_b)
                x_copy.append(x_transformed)
            outputs.append(torch.cat(x_copy, dim=0))

        return torch.cat(outputs, dim=0)

    def transform(self, x, saliency_list=None, **kwargs):
        """
        可在外部传入 transform(x, saliency_list=xxx) 使用显著性图，也可不传，使用默认逻辑
        """
        return self.transform_with_saliency(x, saliency_list=saliency_list)

    def get_loss(self, logits, label):
        """
        计算 SIA 的损失，和原有逻辑保持一致
        """
        return (
            -self.loss(logits, label.repeat(self.num_scale))
            if self.targeted else
            self.loss(logits, label.repeat(self.num_scale))
        )