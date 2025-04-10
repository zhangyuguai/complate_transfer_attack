import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
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

        # 缓存块索引
        self.current_block_indices = {}

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

        return torch.from_numpy(grayscale_cam).to(self.device)

    def find_salient_box(self, saliency_map):
        """查找显著性图中最重要区域的边界框"""
        batch_size = saliency_map.shape[0]
        h, w = saliency_map.shape[1:]
        boxes = []

        for i in range(batch_size):
            sal_map = saliency_map[i].cpu().numpy()

            # 二值化显著性图
            binary_map = (sal_map > self.saliency_threshold).astype(np.uint8)

            # 使用连通分量标记
            labeled, num_features = ndimage.label(binary_map)

            if num_features == 0:
                # 如果没有显著区域，使用整张图像
                boxes.append((0, 0, h, w))
                continue

            # 查找最大的连通区域
            max_area = 0
            max_box = (0, 0, h, w)

            for label_idx in range(1, num_features + 1):
                component = (labeled == label_idx).astype(np.uint8)

                # 找到该连通区域的边界
                y_indices, x_indices = np.where(component > 0)
                if len(y_indices) == 0:
                    continue

                top, left = y_indices.min(), x_indices.min()
                bottom, right = y_indices.max(), x_indices.max()
                height, width = bottom - top + 1, right - left + 1
                area = height * width

                if area > max_area:
                    max_area = area
                    max_box = (top, left, height, width)

            # 扩展边界框
            top, left, height, width = max_box
            expansion_h = int(height * self.box_expansion)
            expansion_w = int(width * self.box_expansion)

            new_top = max(0, top - expansion_h)
            new_left = max(0, left - expansion_w)
            new_bottom = min(h - 1, top + height + expansion_h)
            new_right = min(w - 1, left + width + expansion_w)

            new_height = new_bottom - new_top + 1
            new_width = new_right - new_left + 1

            # 确保最小框大小
            min_h = int(h * self.min_box_size)
            min_w = int(w * self.min_box_size)

            if new_height < min_h or new_width < min_w:
                center_y = new_top + new_height // 2
                center_x = new_left + new_width // 2

                new_top = max(0, center_y - min_h // 2)
                new_left = max(0, center_x - min_w // 2)
                new_height = min(h - new_top, min_h)
                new_width = min(w - new_left, min_w)

            boxes.append((new_top, new_left, new_height, new_width))

        return boxes

    def get_salient_box(self, x, label):
        """获取显著性边界框和显著性图"""
        # 计算显著性图
        saliency_map = self.compute_saliency_map(x, label)

        # 查找显著区域边界框
        boxes = self.find_salient_box(saliency_map)

        return boxes, saliency_map

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

    def apply_outside_box_noise(self, x, boxes):
        """对边界框外区域应用高斯噪声"""
        batch_size, c, h, w = x.shape
        result = x.clone()

        for i in range(batch_size):
            # 创建掩码，初始全1（表示整个图像）
            mask = torch.ones((h, w), device=self.device)

            # 边界框区域设为0
            top, left, height, width = boxes[i]
            mask[top:top+height, left:left+width] = 0

            # 生成高斯噪声
            noise = torch.randn(c, h, w, device=self.device) * self.outside_noise_strength

            # 仅对边框外区域(mask=1)应用噪声
            result[i] = x[i] + noise * mask.unsqueeze(0)

        return result

    def transform_box_region(self, x, boxes, batch_id):
        """只对边界框内的区域应用分块混洗"""
        batch_size = x.shape[0]
        transformed_images = []

        for i in range(batch_size):
            img = x[i:i+1]
            top, left, height, width = boxes[i]

            # 提取边界框内的区域
            roi = img[:, :, top:top+height, left:left+width]

            # 如果区域太小，直接返回原图像
            if roi.shape[2] <= 4 or roi.shape[3] <= 4:
                transformed_images.append(img)
                continue

            # 对ROI应用块混洗
            transformed_roi = self.box_shuffle(roi)

            # 将变换后的ROI放回原图像
            result = img.clone()
            result[:, :, top:top+height, left:left+width] = transformed_roi

            transformed_images.append(result)

        return torch.cat(transformed_images, dim=0)

    def transform(self, x, label, **kwargs):
        """应用基于显著性边界框的BSR变换"""
        x = x.to(self.device)
        transformed_results = []

        # 获取显著性边界框
        boxes, _ = self.get_salient_box(x, label)

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
        data = data.clone().detach().to(self.device)
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
        momentum = torch.zeros_like(data)

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
            boxes, _ = self.get_salient_box(data + delta.detach(), label)

            # 平衡框内外扰动
            delta.data = self.balanced_perturbation(data, delta.data, boxes)

            # 确保原始图像+扰动在有效范围内 [0, 1]
            delta.data = torch.clamp(data + delta.data, 0, 1) - data

        return delta.detach()


    def project(self, delta, epsilon, norm):
        """将扰动投影到范数球内"""
        if norm == 'linfty':
            delta = torch.clamp(delta, -epsilon, epsilon)
        elif norm == 'l2':
            delta_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
            mask = (delta_norm > epsilon).float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            scaling = epsilon / (delta_norm + 1e-12)
            scaling = scaling.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            delta = delta * scaling * mask + delta * (1. - mask)
        return delta
