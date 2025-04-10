import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
from scipy import ndimage
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.onnx.symbolic_opset9 import detach
from tqdm import tqdm
plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

from .utils import *
from .gradient.mifgsm import MIFGSM

class BSR(MIFGSM):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'(https://https://arxiv.org/abs/2308.10299)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of shuffled copies in each iteration.
        num_block (int): the number of block in the image.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bsr/resnet18 --attack bsr --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/bsr/resnet18 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=20, num_block=2,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy',
                 saliency_threshold=0.5, box_expansion=0.1, min_box_size=0.1, outside_noise_strength=0.03,
                 inside_epsilon=None, outside_epsilon=None, device=None, attack='BSR', **kwargs):

        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        # BSR原有参数
        self.num_scale = num_scale
        self.num_block = num_block

        # 显著性相关参数
        self.saliency_threshold = saliency_threshold
        self.box_expansion = box_expansion
        self.min_box_size = min_box_size
        self.outside_noise_strength = outside_noise_strength

        # 框内外扰动分配
        self.inside_epsilon = inside_epsilon if inside_epsilon is not None else epsilon
        self.outside_epsilon = outside_epsilon if outside_epsilon is not None else epsilon * 0.5

        # 初始化显著性检测器
        self.cam_model = None
        self.target_layers = None


        print(f"初始化显著性引导的 {attack} 攻击，分块数={num_block}, 混洗副本数={num_scale}")
        print(f"显著性阈值={saliency_threshold}, 框内扰动={self.inside_epsilon}, 框外扰动={self.outside_epsilon}")
    #进行尝试性


    def get_salient_box(self, x, label):
        """获取显著性图中的重要区域，并创建以最显著点为圆心的圆形区域"""
        # 计算显著性图
        saliency_map = self.compute_saliency_map(x,label)
        #展示一下多尺度的显著性图和单吃土temp显著性图的
        # 展示多尺度和单尺度显著性图对比
        batch_size = x.shape[0]
        # if batch_size > 0:  # 确保至少有一个样本
        #     plt.figure(figsize=(15, 4 * min(batch_size, 2)))
        #     for i in range(min(batch_size, 2)):  # 最多展示前两个样本
        #         # 显示原始图像
        #         plt.subplot(min(batch_size, 2), 3, i*3 + 1)
        #         plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
        #         plt.title(f'样本 {i} 原始图像')
        #         plt.axis('off')
        #
        #         # 显示单尺度显著性图
        #         plt.subplot(min(batch_size, 2), 3, i*3 + 2)
        #         plt.imshow(temp[i].detach().cpu().numpy(), cmap='jet')
        #         plt.colorbar()
        #         plt.title(f'样本 {i} 单尺度显著性图')
        #         plt.axis('off')
        #
        #         # 显示多尺度显著性图
        #         plt.subplot(min(batch_size, 2), 3, i*3 + 3)
        #         plt.imshow(saliency_map[i].detach().cpu().numpy(), cmap='jet')
        #         plt.colorbar()
        #         plt.title(f'样本 {i} 多尺度显著性图')
        #         plt.axis('off')
        #
        #     plt.tight_layout()
        #     plt.show()
        #展示一下获取的显著性图
        # batch_size = x.shape[0]
        # if batch_size > 0:  # 确保至少有一个样本
        #     plt.figure(figsize=(12, 4 * min(batch_size, 2)))
        #     for i in range(min(batch_size, 2)):  # 最多展示前两个样本
        #         # 显示原始图像
        #         plt.subplot(min(batch_size, 2), 2, i*2 + 1)
        #         plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
        #         plt.title(f'样本 {i} 原始图像')
        #         plt.axis('off')
        #
        #         # 显示显著性图
        #         plt.subplot(min(batch_size, 2), 2, i*2 + 2)
        #         plt.imshow(saliency_map[i].detach().cpu().numpy(), cmap='jet')
        #         plt.colorbar()
        #         plt.title(f'样本 {i} 显著性图')
        #         plt.axis('off')
        #
        #     plt.tight_layout()
        #     plt.show()
        # temp = self.compute_saliency_map(x,label)
        batch_size = x.shape[0]
        _, h, w = x.shape[1:]

        # 计算圆形半径（图片长度的1/3）
        radius = int(max(h, w) / 3)
        # radius = 112

        # 存储每个样本的圆形区域信息和掩码
        circle_info = []
        circle_masks = []

        for i in range(batch_size):
            # 获取当前样本的显著性图
            curr_map = saliency_map[i]
            #显示一下当前样本的显著性图
            # plt.figure()
            # plt.imshow(curr_map.cpu().numpy(), cmap='jet')
            # plt.colorbar()
            # plt.title(f'样本 {i} 的显著性图')
            # plt.show()
            # 找到显著性最高的点作为圆心
            flat_idx = torch.argmax(curr_map)
            center_y = int(flat_idx.item() // curr_map.shape[1])
            center_x = int(flat_idx.item() % curr_map.shape[1])

            # 创建圆形掩码
            y_grid, x_grid = torch.meshgrid(torch.arange(h, device=self.device),
                                            torch.arange(w, device=self.device),
                                            indexing='ij')
            dist_from_center = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
            circle_mask = (dist_from_center <= radius).float()

            # 存储圆形信息和掩码
            circle_info.append((center_y, center_x, radius))
            circle_masks.append(circle_mask)
        # 展示一下样本的显著性图和圆形区域
        # for i in range(batch_size):
        #     plt.figure()
        #     plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
        #     circle = plt.Circle((circle_info[i][1], circle_info[i][0]), circle_info[i][2], color='r', fill=False)
        #     plt.gca().add_patch(circle)
        #     plt.title(f'Sample {i} with Salient Circle')
        #     plt.show()


        # 将掩码堆叠成批次
        circle_masks = torch.stack(circle_masks)

        return circle_info, circle_masks, saliency_map
    def compute_saliency_map(self, x, label):
        """使用GradCAM计算显著性图"""
        # 延迟初始化CAM模型
        if self.cam_model is None:
            # 根据模型结构选择合适的目标层
            # 这里假设使用了类似Inception的模型
            # self.target_layers = [self.model[1].Mixed_7c.branch_pool.conv]
            self.target_layers = [self.model[1].layer4[-1]]
            print(self.model[1].layer4[-1].conv2)
            # 创建GradCAM实例
            self.cam_model = GradCAM(
                model=self.model,
                target_layers=self.target_layers,
                use_cuda=self.device.type == 'cuda'
            )

        # 准备GradCAM的目标
        batch_size = x.shape[0]
        targets = [ClassifierOutputTarget(label[i]) for i in range(batch_size)]

        # 生成显著性图
        grayscale_cam = self.cam_model(input_tensor=x, targets=targets)

        return torch.from_numpy(grayscale_cam).to(self.device)
    def get_length(self, length):
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)
    # def transform_box_region(self, x, circle_data, batch_id):
    #     """在圆形区域内应用多样化变换，圆形区域外保持不变"""
    #     circle_info, circle_masks, _ = circle_data
    #     batch_size = x.shape[0]
    #     transformed_images = []
    #
    #     # 定义可用的变换操作列表 - 每次随机选择不同的变换
    #     transform_ops = [
    #         self.apply_random_rotation,      # 随机旋转变换
    #         self.apply_random_noise,         # 添加随机噪声
    #         self.apply_random_color_jitter,  # 颜色抖动
    #         self.apply_random_perspective,   # 透视变换
    #         self.apply_elastic_transform,    # 弹性变换
    #         self.apply_random_blur           # 高斯模糊
    #     ]
    #
    #     # 每个batch_id使用不同的变换组合，增加多样性
    #     random.seed(batch_id)  # 确保每个batch_id的变换是固定的
    #     selected_transforms = random.sample(transform_ops, k=min(3, len(transform_ops)))
    #
    #     for i in range(batch_size):
    #         img = x[i:i+1]  # 保持4D形状 [1,C,H,W]
    #         center_y, center_x, radius = circle_info[i]
    #         mask = circle_masks[i]
    #
    #         # 创建边界框以包含圆形区域
    #         top = max(0, center_y - radius)
    #         left = max(0, center_x - radius)
    #         bottom = min(img.shape[2] - 1, center_y + radius)
    #         right = min(img.shape[3] - 1, center_x + radius)
    #
    #         height = bottom - top + 1
    #         width = right - left + 1
    #
    #         # 如果区域太小，直接返回原图像
    #         if height <= 4 or width <= 4:
    #             transformed_images.append(img)
    #             continue
    #
    #         # 提取包含圆形的矩形区域
    #         roi = img[:, :, top:top+height, left:left+width]
    #
    #         # 创建该区域内的圆形掩码
    #         local_mask = mask[top:top+height, left:left+width].unsqueeze(0).unsqueeze(0)
    #
    #         # 保存原始ROI用于混合
    #         original_roi = roi.clone()
    #         transformed_roi = roi.clone()
    #
    #         # 对ROI应用选定的随机变换
    #         for transform_op in selected_transforms:
    #             transformed_roi = transform_op(transformed_roi)
    #
    #         # 只在圆形区域内应用变换，圆形外保持原样
    #         blended_roi = transformed_roi * local_mask + original_roi * (1 - local_mask)
    #
    #         # 将变换后的ROI放回原图像
    #         result = img.clone()
    #         result[:, :, top:top+height, left:left+width] = blended_roi
    #         #展示一下变换后的结果
    #         # if i == 0:  # 只显示第一个样本以避免过多图像
    #         #     plt.figure(figsize=(12, 4))
    #         #     plt.subplot(1, 3, 1)
    #         #     plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
    #         #     plt.title('原始图像')
    #         #     plt.axis('off')
    #         #
    #         #     plt.subplot(1, 3, 2)
    #         #     plt.imshow(original_roi[0].permute(1, 2, 0).detach().cpu().numpy())
    #         #     plt.title('原始ROI区域')
    #         #     plt.axis('off')
    #         #
    #         #     plt.subplot(1, 3, 3)
    #         #     plt.imshow(result[0].permute(1, 2, 0).detach().cpu().numpy())
    #         #     plt.title('变换后的结果')
    #         #     circle = plt.Circle((center_x - left, center_y - top), radius, color='r', fill=False)
    #         #     plt.gca().add_patch(circle)
    #         #     plt.axis('off')
    #         #
    #         #     plt.tight_layout()
    #         #     plt.show()
    #         transformed_images.append(result)
    #
    #     return torch.cat(transformed_images, dim=0)
    def transform_box_region(self, x, circle_data, batch_id):
        """在随机偏移的圆形区域内应用多样化变换，增强迁移攻击效果"""
        circle_info, circle_masks, saliency_maps = circle_data
        batch_size = x.shape[0]
        transformed_images = []

        # 定义可用的变换操作列表
        transform_ops = [
            self.apply_random_rotation,      # 随机旋转变换
            self.apply_random_noise,         # 添加随机噪声
            self.apply_random_color_jitter,  # 颜色抖动
            self.apply_random_perspective,   # 透视变换
            self.apply_elastic_transform,    # 弹性变换
            self.apply_random_blur           # 高斯模糊
        ]

        # 每个batch_id使用不同的变换组合和偏移策略
        random.seed(batch_id)
        selected_transforms = random.sample(transform_ops, k=min(3, len(transform_ops)))

        # 偏移量系数 - 随不同批次变化
        offset_ratio = 0.15 + 0.1 * random.random()  # 偏移量为半径的15%~25%

        for i in range(batch_size):
            img = x[i:i+1]  # 保持4D形状 [1,C,H,W]
            center_y, center_x, radius = circle_info[i]
            mask = circle_masks[i]

            # 计算基于半径的随机偏移量
            max_offset = int(radius * offset_ratio)
            offset_y = random.randint(-max_offset, max_offset)
            offset_x = random.randint(-max_offset, max_offset)

            # 应用偏移，同时确保圆形区域仍在图像内
            new_center_y = min(max(center_y + offset_y, radius), img.shape[2] - radius - 1)
            new_center_x = min(max(center_x + offset_x, radius), img.shape[3] - radius - 1)

            # 创建新的圆形掩码
            h, w = img.shape[2], img.shape[3]
            y_grid, x_grid = torch.meshgrid(
                torch.arange(h, device=self.device),
                torch.arange(w, device=self.device),
                indexing='ij'
            )
            dist_from_center = torch.sqrt((y_grid - new_center_y)**2 + (x_grid - new_center_x)**2)
            new_mask = (dist_from_center <= radius).float()

            # 创建边界框以包含新的圆形区域
            top = max(0, new_center_y - radius)
            left = max(0, new_center_x - radius)
            bottom = min(img.shape[2] - 1, new_center_y + radius)
            right = min(img.shape[3] - 1, new_center_x + radius)

            height = bottom - top + 1
            width = right - left + 1

            # 如果区域太小，直接返回原图像
            if height <= 4 or width <= 4:
                transformed_images.append(img)
                continue

            # 提取包含圆形的矩形区域
            roi = img[:, :, top:top+height, left:left+width]

            # 创建该区域内的圆形掩码
            local_mask = new_mask[top:top+height, left:left+width].unsqueeze(0).unsqueeze(0)

            # 保存原始ROI用于混合
            original_roi = roi.clone()
            transformed_roi = roi.clone()

            # 对ROI应用选定的随机变换
            for transform_op in selected_transforms:
                transformed_roi = transform_op(transformed_roi)

            # 只在圆形区域内应用变换，圆形外保持原样
            blended_roi = transformed_roi * local_mask + original_roi * (1 - local_mask)

            # 将变换后的ROI放回原图像
            result = img.clone()
            result[:, :, top:top+height, left:left+width] = blended_roi

            # 可视化（仅对部分样本）
            if i == 0 and batch_id % 5 == 0:  # 只对少数批次的第一个样本进行可视化
                plt.figure(figsize=(15, 5))

                # 原始图像和最初圆形
                plt.subplot(1, 3, 1)
                plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
                original_circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
                plt.gca().add_patch(original_circle)
                plt.title(f'原始图像与中心区域')
                plt.axis('off')

                # 偏移后的圆形
                plt.subplot(1, 3, 2)
                plt.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy())
                new_circle = plt.Circle((new_center_x, new_center_y), radius, color='g', fill=False)
                plt.gca().add_patch(new_circle)
                plt.title(f'偏移后的目标区域\n偏移量: ({offset_x}, {offset_y})')
                plt.axis('off')

                # 变换后的结果
                plt.subplot(1, 3, 3)
                plt.imshow(result[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.title('变换后的结果图像')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

            transformed_images.append(result)

        return torch.cat(transformed_images, dim=0)
    def apply_random_rotation(self, x):
        """对输入应用随机旋转"""
        rotation_transform = T.RandomRotation(
            degrees=(-30, 30),
            interpolation=T.InterpolationMode.BILINEAR
        )
        return rotation_transform(x)

    def apply_random_noise(self, x, noise_std=0.05):
        """添加随机高斯噪声"""
        noise = torch.randn_like(x) * noise_std
        return torch.clamp(x + noise, 0, 1)

    def apply_random_color_jitter(self, x):
        """对输入应用随机颜色变换"""
        color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
        return color_jitter(x)

    def apply_random_perspective(self, x, distortion_scale=0.2):
        """对输入应用随机透视变换"""
        perspective_transform = T.RandomPerspective(
            distortion_scale=distortion_scale,
            p=1.0
        )
        return perspective_transform(x)

    def apply_elastic_transform(self, x, alpha=50, sigma=5):
        """应用弹性变换 - 模拟物体变形"""
        # 获取形状信息
        _, c, h, w = x.shape

        # 创建随机位移场
        dx = torch.randn(h, w).to(x.device) * alpha
        dy = torch.randn(h, w).to(x.device) * alpha

        # 高斯滤波使位移场更平滑
        dx = torch.from_numpy(
            ndimage.gaussian_filter(dx.cpu().numpy(), sigma)
        ).to(x.device)
        dy = torch.from_numpy(
            ndimage.gaussian_filter(dy.cpu().numpy(), sigma)
        ).to(x.device)

        # 创建网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing='ij'
        )

        # 应用位移
        x_map = x_grid + dx
        y_map = y_grid + dy

        # 归一化到[-1,1]范围
        x_map = 2 * x_map / (w - 1) - 1
        y_map = 2 * y_map / (h - 1) - 1

        # 堆叠成采样网格
        grid = torch.stack([x_map, y_map], dim=-1).unsqueeze(0)

        # 应用网格采样
        result = F.grid_sample(
            x, grid, mode='bilinear', padding_mode='reflection', align_corners=True
        )

        return result

    def apply_random_blur(self, x, kernel_size=5):
        """应用高斯模糊"""
        blur = T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        return blur(x)
    # def get_length(self, length):
    #     """生成随机分块长度，确保每个块至少有一定的最小长度"""
    #     min_size = max(1, length // (self.num_block * 2))
    #     max_blocks = length // min_size
    #     actual_blocks = min(self.num_block, max_blocks)

    #     if actual_blocks <= 1:
    #         return (length,)

    #     rand = torch.rand(actual_blocks, device=self.device)
    #     remaining_length = length - (min_size * actual_blocks)
    #     extra_lengths = (rand / rand.sum() * remaining_length).round().int()
    #     block_lengths = torch.full((actual_blocks,), min_size, device=self.device) + extra_lengths
    #     block_lengths[-1] += (length - block_lengths.sum())

    #     return tuple(block_lengths.tolist())
    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        rotation_transform = T.RandomRotation(degrees=(-24, 24), interpolation=T.InterpolationMode.BILINEAR)
        return  rotation_transform(x)

    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(self.image_rotation(x_strip), dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    # def transform(self, x, **kwargs):
    #     """
    #     Scale the input for BSR
    #     """
    #     return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])



    def transform(self, x, label, **kwargs):
        """应用基于显著性边界框的BSR变换"""
        x = x.to(self.device)
        transformed_results = []

        #将输入图像拆分为多个不同尺度（Scale）或分块（Patch），分别对这些块进行 GradCAM 计算，再进行拼接或融合。
        # 获取显著性边界框
        boxes = self.get_salient_box(x, label)
        # 实现多尺度GradCAM计算
        #multi_scale_saliency = self.compute_multi_scale_saliency(x, label)
        for i in range(self.num_scale):
            # 对边界框内区域应用分块混洗
            transformed = self.transform_box_region(x, boxes, batch_id=i)


            transformed_results.append(transformed)

        # 连接所有结果
        return torch.cat(transformed_results, dim=0)

    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)


        #初始化扰动
        delta = self.init_delta(data)

        #初始化动量
        momentum = 0
        for _ in tqdm(range(self.epoch), desc=f"Attack: {self.attack} attack_model:  {self.model_name}"):

            #获取输出
            logits = self.get_logits(self.transform(data+delta,label))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))



