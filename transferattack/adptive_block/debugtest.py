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

class SaliencyGuidedBSR(MIFGSM):
    """
    Saliency-Guided BSR Attack

    该方法通过显著性图识别模型决策中最重要的区域，然后有针对性地对框内外区域进行不同强度和类型的扰动。

    Arguments:
        model_name (str): 代理模型名称.
        epsilon (float): 最大扰动预算.
        alpha (float): 每步扰动大小.
        epoch (int): 迭代次数.
        decay (float): 动量衰减系数.
        num_scale (int): 每次迭代中的混洗副本数量.
        num_block (int): 图像分块数量.
        targeted (bool): 是否为目标攻击.
        random_start (bool): 是否随机初始化扰动.
        norm (str): 扰动范数类型, l2/linfty.
        loss (str): 损失函数类型.
        saliency_threshold (float): 显著性阈值，用于确定重要区域.
        box_expansion (float): 边界框扩展比例.
        min_box_size (float): 最小边界框大小（图像尺寸的比例）.
        outside_noise_strength (float): 框外噪声强度.
        inside_epsilon (float): 框内区域的扰动预算.
        outside_epsilon (float): 框外区域的扰动预算.
        device (torch.device): 计算设备.
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=20, num_block=2,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy',
                 saliency_threshold=0.3, box_expansion=0.1, min_box_size=0.1, outside_noise_strength=0.03,
                 inside_epsilon=None, outside_epsilon=None, device=None, attack='SaliencyGuidedBSR', **kwargs):

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

    def compute_saliency_map(self, x, label):
        """使用GradCAM计算显著性图"""
        # 延迟初始化CAM模型
        if self.cam_model is None:
            # 根据模型结构选择合适的目标层
            # 这里假设使用了类似Inception的模型
            self.target_layers = [self.model[1].Mixed_7c.branch_pool.conv]

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
        # 叠加显著性图到原始图片上
        # grayscale_cam = grayscale_cam[0]
        # #报错 Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        #
        # rgb_img = x[0].permute(1, 2, 0).detach().cpu().numpy()
        # rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        # overlay = np.uint8(255 * grayscale_cam)
        # overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        # overlay = cv2.addWeighted(overlay, 0.5, np.uint8(255 * rgb_img), 0.5, 0)
        # plt.imshow(overlay)
        # plt.title('Saliency Map Overlay')
        # plt.show()

        return torch.from_numpy(grayscale_cam).to(self.device)

    # def find_salient_box(self, saliency_map):
    #     """查找显著性图中最重要区域的边界框"""
    #     batch_size = saliency_map.shape[0]
    #     h, w = saliency_map.shape[1:]
    #     boxes = []
    #
    #     for i in range(batch_size):
    #         sal_map = saliency_map[i].cpu().numpy()
    #
    #         # 二值化显著性图
    #         binary_map = (sal_map > self.saliency_threshold).astype(np.uint8)
    #
    #         # 使用连通分量标记
    #         labeled, num_features = ndimage.label(binary_map)
    #
    #         if num_features == 0:
    #             # 如果没有显著区域，使用整张图像
    #             boxes.append((0, 0, h, w))
    #             continue
    #
    #         # 查找最大的连通区域
    #         max_area = 0
    #         max_box = (0, 0, h, w)
    #
    #         for label_idx in range(1, num_features + 1):
    #             component = (labeled == label_idx).astype(np.uint8)
    #
    #             # 找到该连通区域的边界
    #             y_indices, x_indices = np.where(component > 0)
    #             if len(y_indices) == 0:
    #                 continue
    #
    #             top, left = y_indices.min(), x_indices.min()
    #             bottom, right = y_indices.max(), x_indices.max()
    #             height, width = bottom - top + 1, right - left + 1
    #             area = height * width
    #
    #             if area > max_area:
    #                 max_area = area
    #                 max_box = (top, left, height, width)
    #
    #         # 扩展边界框
    #         top, left, height, width = max_box
    #         expansion_h = int(height * self.box_expansion)
    #         expansion_w = int(width * self.box_expansion)
    #
    #         new_top = max(0, top - expansion_h)
    #         new_left = max(0, left - expansion_w)
    #         new_bottom = min(h - 1, top + height + expansion_h)
    #         new_right = min(w - 1, left + width + expansion_w)
    #
    #         new_height = new_bottom - new_top + 1
    #         new_width = new_right - new_left + 1
    #
    #         # 确保最小框大小
    #         min_h = int(h * self.min_box_size)
    #         min_w = int(w * self.min_box_size)
    #
    #         if new_height < min_h or new_width < min_w:
    #             center_y = new_top + new_height // 2
    #             center_x = new_left + new_width // 2
    #
    #             new_top = max(0, center_y - min_h // 2)
    #             new_left = max(0, center_x - min_w // 2)
    #             new_height = min(h - new_top, min_h)
    #             new_width = min(w - new_left, min_w)
    #
    #         boxes.append((new_top, new_left, new_height, new_width))
    #
    #     return boxes

    # def get_salient_box(self, x, label):
    #     """获取显著性图中的重要区域，并创建以最显著点为圆心的圆形区域"""
    #     # 计算显著性图
    #     saliency_map = self.compute_saliency_map(x, label)
    #
    #     batch_size = x.shape[0]
    #     _, h, w = x.shape[1:]
    #
    #     # 计算圆形半径（图片长度的1/3）
    #     radius = int(max(h, w) / 2)
    #
    #     # 存储每个样本的圆形区域信息和掩码
    #     circle_info = []
    #     circle_masks = []
    #
    #     for i in range(batch_size):
    #         # 获取当前样本的显著性图
    #         curr_map = saliency_map[i]
    #
    #         # 找到显著性最高的点作为圆心
    #         flat_idx = torch.argmax(curr_map)
    #         center_y = int(flat_idx.item() // curr_map.shape[1])
    #         center_x = int(flat_idx.item() % curr_map.shape[1])
    #
    #         # 创建圆形掩码
    #         y_grid, x_grid = torch.meshgrid(torch.arange(h, device=self.device),
    #                                        torch.arange(w, device=self.device),
    #                                        indexing='ij')
    #         dist_from_center = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
    #         circle_mask = (dist_from_center <= radius).float()
    #
    #         # 存储圆形信息和掩码
    #         circle_info.append((center_y, center_x, radius))
    #         circle_masks.append(circle_mask)
    #     # 展示一下样本的显著性图和圆形区域
    #     for i in range(batch_size):
    #         plt.figure()
    #         plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
    #         circle = plt.Circle((circle_info[i][1], circle_info[i][0]), circle_info[i][2], color='r', fill=False)
    #         plt.gca().add_patch(circle)
    #         plt.title(f'Sample {i} with Salient Circle')
    #         plt.show()
    #
    #
    #     # 将掩码堆叠成批次
    #     circle_masks = torch.stack(circle_masks)
    #
    #     return circle_info, circle_masks, saliency_map
    def get_salient_box(self, x, label):
        """获取多个显著性区域，以每个区域中心为圆心创建圆形区域"""
        # 计算显著性图
        saliency_map = self.compute_saliency_map(x, label)

        batch_size = x.shape[0]
        _, h, w = x.shape[1:]

        # 默认半径（可根据需要调整）
        radius = int(min(h, w) * 0.15)  # 使用图像较小边的15%作为默认半径

        # 存储每个样本的圆形区域信息和掩码
        all_circle_infos = []
        all_circle_masks = []

        for i in range(batch_size):
            # 获取当前样本的显著性图
            curr_map = saliency_map[i]

            # 二值化显著性图，找出所有高于阈值的区域
            binary_map = (curr_map > self.saliency_threshold).cpu().numpy().astype(np.uint8)

            # 标记连通区��
            labeled, num_features = ndimage.label(binary_map)

            if num_features == 0:
                # 如果没有显著区域，使用图像中心
                center_y, center_x = h // 2, w // 2
                centers = [(center_y, center_x)]
            else:
                centers = []
                # 计算每个连通区域的中心
                for region_idx in range(1, num_features + 1):
                    y_indices, x_indices = np.where(labeled == region_idx)
                    if len(y_indices) > 0:
                        center_y = int(y_indices.mean())
                        center_x = int(x_indices.mean())
                        centers.append((center_y, center_x))

            # 创建组合掩码，将所有圆���区域合并
            combined_mask = torch.zeros((h, w), device=self.device)
            circle_infos = []

            # 为每个中心创建圆形区域
            for center_y, center_x in centers:
                # 创建圆形掩码
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(h, device=self.device),
                    torch.arange(w, device=self.device),
                    indexing='ij'
                )
                dist_from_center = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
                circle_mask = (dist_from_center <= radius).float()

                # 将该圆形区域添加到组合掩码
                combined_mask = torch.clamp(combined_mask + circle_mask, 0, 1)

                # 保存圆形信息
                circle_infos.append((center_y, center_x, radius))

            # 存储该样本的所有圆形信息和组合掩码
            all_circle_infos.append(circle_infos)
            all_circle_masks.append(combined_mask)

            # 可视化显著性区域和圆形掩码（用于调试）
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.title(f'原始图像 {i}')

            plt.subplot(1, 3, 2)
            plt.imshow(curr_map.cpu().numpy(), cmap='jet')
            plt.title(f'显著性图 {i}')

            plt.subplot(1, 3, 3)
            plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
            for center_y, center_x, r in circle_infos:
                circle = plt.Circle((center_x, center_y), r, color='r', fill=False)
                plt.gca().add_patch(circle)
            plt.title(f'显著性区域圆形标记 {i}')
            plt.show()

        return all_circle_infos, all_circle_masks, saliency_map

    def create_box_masks(self, shape, boxes):
        """创建框内和框外的掩码"""
        batch_size, c, h, w = shape
        inside_masks = []
        outside_masks = []

        for i in range(batch_size):
            # 创建全零掩码
            inside_mask = torch.zeros((h, w), device=self.device)

            # 边界框区域设为1
            top, left, height, width = boxes[i]
            inside_mask[top:top+height, left:left+width] = 1

            # 框外掩码 = 1 - 框内掩码
            outside_mask = 1 - inside_mask

            # 扩展通道维度
            inside_masks.append(inside_mask.unsqueeze(0).repeat(c, 1, 1))
            outside_masks.append(outside_mask.unsqueeze(0).repeat(c, 1, 1))

        return torch.stack(inside_masks), torch.stack(outside_masks)

    def get_length(self, length):
        """生成随机块长度"""
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        """单一维度的分块混洗"""
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def box_shuffle(self, x):
        """对框内区域进行分块混洗(无旋转)"""
        dims = [2, 3]  # 高度和宽度维度
        random.shuffle(dims)

        # 第一维度分块
        x_strips = self.shuffle_single_dim(x, dims[0])

        # 第二维度分块并连接
        result = torch.cat([
            torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1])
            for x_strip in x_strips
        ], dim=dims[0])

        return result

    def apply_outside_box_noise(self, x, circle_data):
        """对圆形区域外应用像素缩放"""
        # 解析圆形区域数据
        circle_info, circle_masks, _ = circle_data

        batch_size, c, h, w = x.shape
        result = x.clone()

        # 像素缩放因子（可以作为参数添加到类初始化中）
        scale_factor = 0.8  # 缩小到原来的80%

        for i in range(batch_size):
            # 获取圆形掩码（圆内为1，圆外为0）
            mask = circle_masks[i]

            # 反转掩码，使圆外为1，圆内为0
            outside_mask = 1 - mask

            # 对圆外区域应用像素缩放
            for ch in range(c):
                # 在圆形区域内保持原始像素值，在圆形区域外应用缩放
                result[i, ch] = x[i, ch] * mask + (x[i, ch] * scale_factor) * outside_mask

            # 可视化处理结果（调试用）
            if i == 0:  # 只展示第一个样本
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(x[i].permute(1, 2, 0).detach().cpu().numpy())
                plt.title('原始图像')
                plt.subplot(1, 2, 2)
                plt.imshow(result[i].permute(1, 2, 0).detach().cpu().numpy())
                plt.title('区域外缩放后的图像')
                plt.show()

        return result

    def transform_box_region(self, x, circle_data, batch_id):
        """只对圆形区域内应用分块混洗"""
        circle_info, circle_masks, _ = circle_data

        #展示一下circle_masks
        for idx, mask in enumerate(circle_masks):
            plt.figure()
            plt.imshow(mask.cpu().numpy())
            plt.title(f'Circle Mask {idx}')
            plt.show()
        batch_size = x.shape[0]
        transformed_images = []

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

            # 如果区域太小，直接返回原图像
            if height <= 4 or width <= 4:
                transformed_images.append(img)
                continue

            # 提取包含圆形的矩形区域
            roi = img[:, :, top:top+height, left:left+width]
            # 展示一下roi
            plt.figure()
            plt.imshow(roi[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.title(f'Sample {i} with Circle Region')
            plt.show()

            # 创建该区域内的圆形掩码
            local_mask = mask[top:top+height, left:left+width].unsqueeze(0).unsqueeze(0)

            # 对ROI应用块混洗
            transformed_roi = self.box_shuffle(roi)

            # 只在圆形区域内应用变换，圆形外保持原样
            blended_roi = transformed_roi * local_mask + roi * (1 - local_mask)

            # 将变换后的ROI放回原图像
            result = img.clone()
            result[:, :, top:top+height, left:left+width] = blended_roi
            #展示一下变换后的图像
            plt.figure()
            plt.imshow(result[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.title(f'Sample {i} with Transformed Circle Region')
            plt.show()

            transformed_images.append(result)

        return torch.cat(transformed_images, dim=0)

    def transform(self, x, label, **kwargs):
        """应用基于显著性边界框的BSR变换"""
        x = x.to(self.device)
        transformed_results = []

        # 获取显著性边界框
        boxes = self.get_salient_box(x, label)

        for i in range(self.num_scale):
            # 对边界框内区域应用分块混洗
            transformed = self.transform_box_region(x, boxes, batch_id=i)


            # 对框外区域应用高斯噪声
            transformed = self.apply_outside_box_noise(transformed, boxes)

            transformed_results.append(transformed)

        # 连接所有结果
        return torch.cat(transformed_results, dim=0)

    def balanced_perturbation(self, x, delta, boxes):
        """平衡框内外区域的扰动"""
        # 获取框内外掩码
        inside_masks, outside_masks = self.create_box_masks(x.shape, boxes)

        # 分离框内外扰动
        delta_inside = delta * inside_masks
        delta_outside = delta * outside_masks

        # 分别投影到不同的扰动预算
        delta_inside = torch.clamp(delta, -self.inside_epsilon, self.inside_epsilon)
        delta_outside = torch.clamp(delta, -self.outside_epsilon, self.outside_epsilon)

        # 合并扰动
        balanced_delta = delta_inside + delta_outside

        return balanced_delta

    def forward(self, data, label, **kwargs):
        """执行基于显著性引导的BSR攻击"""
        # 处理目标攻击情况
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        # 确保数据在正确设备上
        data = data.clone().detach().to(self.device) # (N, 3, 224, 224)
        label = label.clone().detach().to(self.device)

        # 初始化扰动
        delta = torch.zeros_like(data, requires_grad=True)
        if self.random_start:
            if self.norm == 'linfty':
                delta.data.uniform_(-self.epsilon, self.epsilon)
            elif self.norm == 'l2':
                delta.data.normal_()
                delta.data = delta.data * self.epsilon / (delta.data.norm(p=2) + 1e-12)

        # 初始化动量
        momentum = torch.zeros_like(data) # (N, 3, 224, 224)

        # 攻击迭代
        for i in tqdm(range(self.epoch)):
            # 清除梯度
            delta.grad = None

            # 对抗样本 = 原图 + 扰动
            x_adv = data + delta

            # 应用基于显著性边界框的BSR变换
            x_transformed = self.transform(x_adv, label)

            # 计算损失和梯度
            outputs = self.model(x_transformed)
            loss = self.get_loss(outputs, label.repeat(self.num_scale))

            loss.backward()
            grad = delta.grad.clone()

            # 更新动量
            momentum = self.get_momentum(grad, momentum)

            # 更新扰动
            delta = self.update_delta(delta, data, momentum, self.alpha)

            # 获取当前边界框
            #boxes = self.get_salient_box(data + delta.detach(), label)

            # 平衡框内外扰动
            #delta.data = self.balanced_perturbation(data, delta.data, boxes)

            # 确保原始图像+扰动在有效范围内 [0, 1]
            delta.data = torch.clamp(data + delta.data, 0, 1) - data

        return delta.detach()


