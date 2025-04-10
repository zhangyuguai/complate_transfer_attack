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


class CrossBSR_Adaptive(CrossBSR):
    def __init__(self, mix_init=0.5, mix_decay=0.95, **kwargs):
        super().__init__(mix_ratio=mix_init, **kwargs)
        self.mix_decay = mix_decay  # 每轮衰减系数

    def update_mix_ratio(self):
        self.mix_ratio *= self.mix_decay  # 逐步降低混合比例

    def forward(self, data, label, **kwargs):
        for epoch in range(self.epoch):
            self.update_mix_ratio()
            delta = super().forward(data, label, **kwargs)
        return delta


class EnhancedCrossBSR(CrossBSR):
    """
    增强版CrossBSR攻击，整合多策略：
    1. 动态混合比例（Adaptive Mix）
    2. 多尺度输入变换（DIM）
    3. 浅层梯度增强（SGM）
    4. 防御模拟（NRDM）
    通过参数开关控制各模块
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
            # 扩展模块参数
            use_dim=True,                # 启用多尺度变换
            dim_resize_range=(0.8, 1.2), # 缩放范围
            use_adaptive_mix=True,       # 启用动态混合
            mix_decay=0.95,              # 混合比例衰减率
            use_sgm=True,                # 启用浅层梯度增强
            sgm_gamma=0.5,              # 浅层梯度权重
            use_defense_sim=False,       # 启用防御模拟
            defense_type='jpeg',         # 防御类型
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

        # 初始化各模块
        self.use_dim = use_dim
        self.dim_resize_range = dim_resize_range
        self.use_adaptive_mix = use_adaptive_mix
        self.mix_decay = mix_decay
        self.use_sgm = use_sgm
        self.sgm_gamma = sgm_gamma
        self.use_defense_sim = use_defense_sim
        self.defense_type = defense_type

        # SGM相关初始化
        if self.use_sgm:
            self._init_sgm_hooks()

    def _init_sgm_hooks(self):
        """注册浅层梯度钩子"""
        self.gradients = []
        self.handles = []

        # 示例：获取ResNet的layer1输出
        for name, module in self.model.named_modules():
            if 'layer1' in name:  # 根据模型结构调整
                handle = module.register_backward_hook(self._save_grad)
                self.handles.append(handle)
                break

    def _save_grad(self, module, grad_input, grad_output):
        """保存浅层梯度"""
        self.gradients.append(grad_output[0].detach())

    def dim_transform(self, x):
        """多尺度变换（支持正/负填充）"""
        resize_factor = random.uniform(*self.dim_resize_range)
        original_h, original_w = x.size(2), x.size(3)
        
        # 计算缩放后尺寸
        new_h = int(original_h * resize_factor)
        new_w = int(original_w * resize_factor)
        
        # 双线性插值调整尺寸
        x_resized = F.interpolate(
            x, size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        
        # 动态处理填充/裁剪逻辑
        def calculate_pad_crop(original, new):
            delta = original - new
            if delta >= 0:
                # 正填充模式
                pad_front = random.randint(0, delta)
                pad_back = delta - pad_front
                return pad_front, pad_back
            else:
                # 负填充（裁剪）模式
                crop_size = -delta
                crop_start = random.randint(0, crop_size)
                return -crop_start, -(crop_size - crop_start)  # 返回负值表示裁剪
        
        # 高度方向处理
        pad_top, pad_bottom = calculate_pad_crop(original_h, new_h)
        # 宽度方向处理
        pad_left, pad_right = calculate_pad_crop(original_w, new_w)
        
        return F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))


    def defense_emulate(self, x):
        """防御模拟"""
        if self.defense_type == 'jpeg':
            return jpeg_compress(x, quality=75)
        elif self.defense_type == 'gaussian':
            return x + torch.randn_like(x) * 0.03
        else:
            return x

    def transform(self, x, **kwargs):
        """集成数据增强流程"""
        # 多尺度变换
        if self.use_dim:
            x = self.dim_transform(x)

        # 跨图像混合
        x_mixed = self.cross_shuffle(x)

        # 防御模拟
        if self.use_defense_sim:
            x_mixed = self.defense_emulate(x_mixed)

        # BSR标准变换
        return torch.cat([self.shuffle(x_mixed) for _ in range(self.num_scale)], dim=0)

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



    def update_mix_ratio(self):
        """动态调整混合比例"""
        if self.use_adaptive_mix:
            self.mix_ratio = max(0.01, self.mix_ratio * self.mix_decay)

    def forward(self, data, label, **kwargs):
        # 迭代过程动态调整
        for _ in range(self.epoch):
            self.update_mix_ratio()
            delta = super().forward(data, label, **kwargs)
        return delta

    def __del__(self):
        # 清理钩子
        for handle in self.handles:
            handle.remove()

import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from ..utils import *
from ..gradient.mifgsm import MIFGSM

class EnhancedBSR(MIFGSM):
    """
    Enhanced BSR Attack with:
    - Fixed 2x2 block partition (as original paper)
    - Dynamic cross-image mixing
    - Frequency-domain gradient calibration
    - Multi-scale integration
    - Attention consistency constraint
    """

    def __init__(
            self,
            model_name,
            epsilon=16/255,
            alpha=1.6/255,
            epoch=10,
            decay=1.0,
            num_scale=20,
            mix_range=(0.1, 0.5),
            dim_scales=[0.8, 1.0, 1.2],
            attn_weight=0.3,
            phase_noise=0.1,
            targeted=False,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=None,
            attack='EnhancedBSR',
            **kwargs
    ):
        super().__init__(
            model_name, epsilon, alpha, epoch, decay,
            targeted, random_start, norm, loss, device, attack
        )

        # Attack parameters
        self.num_scale = num_scale
        self.mix_range = mix_range
        self.dim_scales = dim_scales
        self.attn_weight = attn_weight
        self.phase_noise = phase_noise

        # Transformation modules
        self.rotation_transform = T.RandomRotation(
            degrees=(-24, 24),
            interpolation=T.InterpolationMode.BILINEAR
        )

        # Attention storage
        self.attention_maps = None

    def _register_hooks(self):
        """Register hooks for Grad-CAM computation"""
        self.feature_maps = []
        self.gradients = []

        def forward_hook(module, input, output):
            self.feature_maps.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        # Register hooks to last convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and 'layer4' in name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def compute_gradcam(self, x, target):
        """Compute Grad-CAM attention maps"""
        self.feature_maps = []
        self.gradients = []

        # Forward pass
        logits = self.model(x)
        score = logits[:, target]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Get feature maps and gradients
        features = self.feature_maps[-1]
        grads = self.gradients[-1]

        # Pool gradients
        pooled_grads = torch.mean(grads, dim=[2,3], keepdim=True)
        return torch.relu(torch.sum(features * pooled_grads, dim=1, keepdim=True))

    def fixed_block_partition(self, x):
        """Fixed 2x2 block partition with equal size"""
        b, c, h, w = x.shape
        return [
            x[:, :, :h//2, :w//2],  # Top-left
            x[:, :, :h//2, w//2:],  # Top-right
            x[:, :, h//2:, :w//2],  # Bottom-left
            x[:, :, h//2:, w//2:],  # Bottom-right
        ]

    def shuffle(self, x):
        """Enhanced shuffle with rotation and fixed partitioning"""
        blocks = self.fixed_block_partition(x)
        random.shuffle(blocks)

        # Apply rotation and reassemble
        rotated_blocks = [self.rotation_transform(b) for b in blocks]

        top = torch.cat(rotated_blocks[:2], dim=3)
        bottom = torch.cat(rotated_blocks[2:], dim=3)
        return torch.cat([top, bottom], dim=2)

    def dynamic_mixing(self, x, epoch):
        """Dynamic cross-image mixing with cosine annealing"""
        mix_ratio = self.mix_range[0] + 0.5*(self.mix_range[1]-self.mix_range[0])*(1+np.cos(epoch*np.pi/self.epoch))

        mixed = []
        for img in x:
            # Randomly select another image
            mix_img = x[random.randint(0, len(x)-1)]

            # Create binary mask for mixing
            mask = torch.rand_like(img) < mix_ratio
            mixed.append(torch.where(mask, mix_img, img))

        return torch.stack(mixed)

    def multi_scale_transform(self, x):
        """Multi-scale integration"""
        scaled_images = []
        for s in self.dim_scales:
            h = int(x.size(2)*s)
            w = int(x.size(3)*s)
            scaled = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=False)
            scaled_images.append(F.interpolate(scaled, size=x.shape[2:], mode='bilinear'))
        return torch.cat(scaled_images, dim=0)

    def frequency_calibration(self, grad):
        """Frequency-domain gradient calibration"""
        fft_grad = torch.fft.fft2(grad)
        mag = torch.abs(fft_grad)
        phase = torch.angle(fft_grad)

        # Add phase noise
        phase += self.phase_noise * torch.randn_like(phase)

        calibrated = torch.fft.ifft2(mag * torch.exp(1j*phase)).real
        return 0.7*grad + 0.3*calibrated

    def get_attention_loss(self, x_orig, x_adv, target):
        """Attention consistency loss"""
        with torch.enable_grad():
            cam_orig = self.compute_gradcam(x_orig, target)
            cam_adv = self.compute_gradcam(x_adv, target)
        return F.mse_loss(cam_orig, cam_adv)

    def transform(self, x, epoch):
        """Integrated transformation pipeline"""
        # Step 1: Dynamic cross-image mixing
        mixed = self.dynamic_mixing(x, epoch)

        # Step 2: Multi-scale processing
        scaled = self.multi_scale_transform(mixed)

        # Step 3: Shuffle and rotate
        transformed = torch.cat([self.shuffle(scaled) for _ in range(self.num_scale)], dim=0)

        return transformed

    def update(self, delta, data, grad, alpha, **kwargs):
        """Enhanced update with frequency calibration"""
        grad = self.frequency_calibration(grad)
        return super().update(delta, data, grad, alpha, **kwargs)

    def forward(self, data, label, **kwargs):
        self._register_hooks()
        delta = super().forward(data, label, **kwargs)

        # Cleanup hooks
        self.feature_maps = []
        self.gradients = []
        return delta

    def get_loss(self, logits, label, x_orig, x_adv):
        """Loss function with attention constraint"""
        ce_loss = super().get_loss(logits, label)
        attn_loss = self.get_attention_loss(x_orig, x_adv, label)
        return ce_loss + self.attn_weight * attn_loss

    def _forward_iter(self, data, delta, label, epoch, **kwargs):
        # Enhanced forward iteration
        x_orig = data + delta
        x_trans = self.transform(x_orig, epoch)

        # Forward pass
        logits = self.model(x_trans)

        # Compute combined loss
        loss = self.get_loss(logits, label, x_orig, x_trans)
        loss.backward()

        # Gradient calculation
        grad = x_trans.grad.detach()
        grad = self.normalize_grad(grad, self.norm)

        # Update delta
        delta = self.update(delta, data, grad, self.alpha)
        return delta.detach()