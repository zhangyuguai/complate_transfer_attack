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
from tqdm import tqdm

from ..utils import *
from ..gradient.mifgsm import MIFGSM

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
    def transform_roi(self, roi):
        """对ROI应用裁剪、缩放、旋转等变换"""
        # 获取原始ROI尺寸
        _, c, h, w = roi.shape

        # 创建变换列表
        transforms_list = []

        # 随机选择2-3种变换操作
        transform_choices = np.random.choice([
            'crop', 'rotate', 'flip', 'color_jitter', 'perspective'
        ], size=np.random.randint(2, 4), replace=False)

        for choice in transform_choices:
            if choice == 'crop':
                # 随机裁剪和调整大小
                transforms_list.append(
                    T.RandomResizedCrop(
                        size=(h, w),  # 保持输出尺寸与输入相同
                        scale=(0.7, 1.0),
                        ratio=(0.8, 1.2)
                    )
                )

            elif choice == 'rotate':
                # 随机旋转
                transforms_list.append(
                    T.RandomRotation(
                        degrees=(-30, 30),
                        interpolation=T.InterpolationMode.BILINEAR
                    )
                )

            elif choice == 'flip':
                # 随机水平或垂直翻转
                if np.random.random() > 0.5:
                    transforms_list.append(T.RandomHorizontalFlip(p=1.0))
                else:
                    transforms_list.append(T.RandomVerticalFlip(p=1.0))

            elif choice == 'color_jitter':
                # 随机调整亮度、对比度等
                transforms_list.append(
                    T.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1
                    )
                )

            elif choice == 'perspective':
                # 随机透视变换
                transforms_list.append(
                    T.RandomPerspective(
                        distortion_scale=0.2,
                        p=1.0
                    )
                )

        # 组合所有变换
        transform = T.Compose(transforms_list)

        # 应用变换
        transformed_roi = transform(roi)

        return transformed_roi


    def transform_box_region(self, x, circle_data):
        """只对圆形区域内应用分块混洗"""
        circle_info, circle_masks, _ = circle_data
        batch_size = x.shape[0]
        all_transformed_images = []

        for i in range(batch_size):
            img = x[i:i+1]  # 保持4D形状
            center_y, center_x, radius = circle_info[i]
            mask = circle_masks[i]

            # 创建边界框以包含圆形区域
            top = max(0, center_y - radius)
            left = max(0, center_x - radius)
            bottom = min(img.shape[2] - 1, center_y + radius)
            right = min(img.shape[3] - 1, center_x + radius)

            height = bottom - top + 1
            width = right - left + 1

            # 如果区域太小，直接返回原图像复制num_scale次
            if height <= 4 or width <= 4:
                all_transformed_images.append(torch.cat([img] * self.num_scale, dim=0))
                continue

            # 提取包含圆形的矩形区域
            roi = img[:, :, top:top+height, left:left+width]

            # 创建该区域内的圆形掩码
            local_mask = mask[top:top+height, left:left+width].unsqueeze(0).unsqueeze(0)

            # 为当前样本创建num_scale个变换版本
            sample_transformed_images = []
            for _ in range(self.num_scale):
                # 对ROI应用变换
                transformed_roi = self.transform_roi(roi)

                # 只在圆形区域内应用变换，圆形外保持原样
                blended_roi = transformed_roi * local_mask + roi * (1 - local_mask)

                # 将变换后的ROI放回原图像
                result = img.clone()
                result[:, :, top:top+height, left:left+width] = blended_roi
                sample_transformed_images.append(result)

            # 连接当前样本的所有变换版本
            all_transformed_images.append(torch.cat(sample_transformed_images, dim=0))

        # 按批次连接所有样本的变换版本
        return torch.cat(all_transformed_images, dim=0)

    def get_salient_box(self, x, label):
        """获取显著性图中的重要区域，并创建以最显著点为圆心的圆形区域"""
        # 计算显著性图
        saliency_map = self.compute_saliency_map(x, label)

        batch_size = x.shape[0]
        _, h, w = x.shape[1:]

        # 计算圆形半径（图片长度的1/3）
        # radius = int(max(h, w) / 2)
        radius = 112

        # 存储每个样本的圆形区域信息和掩码
        circle_info = []
        circle_masks = []

        for i in range(batch_size):
            # 获取当前样本的显著性图
            curr_map = saliency_map[i]

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
    # def get_salient_box(self, x, label):
    #     """获取显著性图中的重要区域，并创建矩形边界框"""
    #     # 计算显著性图
    #     saliency_map = self.compute_saliency_map(x, label)
    #
    #     batch_size = x.shape[0]
    #     _, h, w = x.shape[1:]
    #
    #     # 矩形默认尺寸
    #     box_width = int(w / 3)  # 图像宽度的三分之一
    #     box_height = int(h / 3)  # 图像高度的三分之一
    #
    #     # 存储每个样本的矩形区域信息和掩码
    #     rect_info = []
    #     rect_masks = []
    #
    #     for i in range(batch_size):
    #         # 获取当前样本的显著性图
    #         curr_map = saliency_map[i]
    #
    #         # 找到显著性最高的点作为矩形中心
    #         flat_idx = torch.argmax(curr_map)
    #         center_y = int(flat_idx.item() // curr_map.shape[1])
    #         center_x = int(flat_idx.item() % curr_map.shape[1])
    #
    #         # 计算矩形边界
    #         top = max(0, center_y - box_height // 2)
    #         left = max(0, center_x - box_width // 2)
    #         bottom = min(h - 1, center_y + box_height // 2)
    #         right = min(w - 1, center_x + box_width // 2)
    #
    #         # 创建矩形掩码
    #         mask = torch.zeros((h, w), device=self.device)
    #         mask[top:bottom+1, left:right+1] = 1.0
    #
    #         # 存储矩形信息和掩码
    #         rect_info.append((top, left, bottom, right))
    #         rect_masks.append(mask)
    #
    #     # 将掩码堆叠成批次
    #     rect_masks = torch.stack(rect_masks)
    #
    #     return rect_info, rect_masks, saliency_map
    #
    # def random_pixel_dropout(self, x, dropout_rate=0.5):
    #     """随机丢弃图像中的像素"""
    #     mask = torch.rand_like(x) > dropout_rate
    #     return x * mask
    #
    # def transform_box_region(self, x, rect_data, batch_id):
    #     """对矩形区域内应用随机像素丢弃"""
    #     rect_info, rect_masks, _ = rect_data
    #     batch_size = x.shape[0]
    #     transformed_images = []
    #
    #     for i in range(batch_size):
    #         img = x[i:i+1]  # 保持4D形状
    #         top, left, bottom, right = rect_info[i]
    #         mask = rect_masks[i]
    #
    #         height = bottom - top + 1
    #         width = right - left + 1
    #
    #         # 如果区域太小，直接返回原图像
    #         if height <= 4 or width <= 4:
    #             transformed_images.append(img)
    #             continue
    #
    #         # 提取矩形区域
    #         roi = img[:, :, top:bottom+1, left:right+1]
    #
    #         # 对ROI应用随机像素丢弃
    #         transformed_roi = self.random_pixel_dropout(roi, dropout_rate=0.3)
    #
    #         # 将变换后的ROI放回原图像
    #         result = img.clone()
    #         result[:, :, top:bottom+1, left:right+1] = transformed_roi
    #
    #         transformed_images.append(result)
    #
    #     return torch.cat(transformed_images, dim=0)
    #
    # def transform(self, x, label, **kwargs):
    #     """应用基于显著性边界框的随机像素丢弃变换"""
    #     x = x.to(self.device)
    #     transformed_results = []
    #
    #     # 获取显著性边界框
    #     boxes = self.get_salient_box(x, label)
    #
    #     for i in range(self.num_scale):
    #         # 对边界框内区域应用随机像素丢弃
    #         transformed = self.transform_box_region(x, boxes, batch_id=i)
    #         #展示一下变换后的图像
    #         plt.figure()
    #         plt.imshow(transformed[0].permute(1, 2, 0).detach().cpu().numpy())
    #         plt.title(f'Sample {i} with Transformed Rectangle Region')
    #         plt.show()
    #
    #
    #         transformed_results.append(transformed)
    #
    #     # 连接所有结果
    #     return torch.cat(transformed_results, dim=0)
    def get_length(self, length):
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

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
    # def transform(self, x, label, **kwargs):
    #     """应用基于显著性边界框的BSR变换"""
    #     x = x.to(self.device)
    #     transformed_results = []
    #
    #     # 获取显著性边界框
    #     boxes = self.get_salient_box(x, label)
    #
    #     for i in range(self.num_scale):
    #         # 对边界框内区域应用分块混洗
    #         transformed = self.transform_box_region(x, boxes, batch_id=i)
    #
    #
    #         transformed_results.append(transformed)
    #
    #     # 连接所有结果
    #     return torch.cat(transformed_results, dim=0)

    # def forward(self, data, label, **kwargs):
    #     data = data.clone().detach().to(self.device)
    #     label = label.clone().detach().to(self.device)
    #
    #
    #     #初始化扰动
    #     delta = self.init_delta(data)
    #     #获取显著性框
    #
    #
    #     #初始化动量
    #     momentum = 0
    #     for _ in tqdm(range(self.epoch), desc=f"Attack: {self.attack} attack_model:  {self.model_name}"):
    #         circle_data = self.get_salient_box(data, label)
    #         #获取输出
    #         logits = self.get_logits(self.transform_box_region(data+delta,circle_data,))
    #
    #         # Calculate the loss
    #         loss = self.get_loss(logits, label)
    #
    #         # Calculate the gradients
    #         grad = self.get_grad(loss, delta)
    #         # Calculate the momentum
    #         momentum = self.get_momentum(grad, momentum)
    #
    #         # Update adversarial perturbation
    #         delta = self.update_delta(delta, data, momentum, self.alpha)
    #
    #
    #
    #     return delta.detach()
    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化扰动
        delta = self.init_delta(data)

        # 获取显著性框 (仅计算一次)
        circle_data = self.get_salient_box(data, label)
        _, circle_masks, _ = circle_data

        # 初始化动量
        momentum = 0
        for _ in tqdm(range(self.epoch), desc=f"Attack: {self.attack} attack_model: {self.model_name}"):
            # 获取输出
            logits = self.get_logits(self.transform_box_region(data+delta, circle_data))

            # 计算损失
            loss = self.get_loss(logits, label)

            # 计算梯度
            grad = self.get_grad(loss, delta)

            # 计算动量
            momentum = self.get_momentum(grad, momentum)

            # 仅对框内区域进行扰动更新
            inside_mask = circle_masks.unsqueeze(1)  # 调整形状以匹配delta: [B, 1, H, W]

            # 使用掩码筛选梯度，仅保留框内区域的梯度
            masked_momentum = momentum * inside_mask

            # 更新扰动 (仅框内区域)
            delta = self.update_delta(delta, data, masked_momentum, self.alpha)

            # 强制框外区域的扰动为零
            delta = delta * inside_mask

        return delta.detach()
    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))

    # 将显著性框的区域与初始显著性框的距离损失加入到损失函数中


