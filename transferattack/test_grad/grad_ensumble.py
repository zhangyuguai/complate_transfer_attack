import torch
import numpy as np
from tqdm import tqdm
from ..attack import Attack

class GradientDiversityAttack(Attack):
    """
    梯度多样性攻击（Gradient Diversity Attack）

    在MI-FGSM的基础上，对梯度方向进行多样化采样，并通过加权聚合形成更有效的攻击方向
    """
    def __init__(self, model_name, epsilon=16/255, targeted=False, random_start=False, norm='linfty', loss='crossentropy',
                 epoch=10, alpha=2/255, decay=1.0, num_directions=10, original_weight=0.6, device=None):
        """
        初始化梯度多样性攻击

        参数:
            model_name (str): 代理模型名称
            epsilon (float): 扰动预算
            targeted (bool): 是否为定向攻击
            random_start (bool): 是否随机初始化delta
            norm (str): 约束范数类型，'l2'或'linfty'
            loss (str): 损失函数类型
            epoch (int): 迭代次数
            alpha (float): 步长
            decay (float): 动量衰减因子
            num_directions (int): 采样的梯度方向数量
            original_weight (float): 原始梯度的权重，应在(0,1)范围内
            device (torch.device): 计算设备
        """
        super(GradientDiversityAttack, self).__init__(
            'GradientDiversityAttack', model_name, epsilon, targeted, random_start, norm, loss, device)
        self.epoch = epoch
        self.alpha = alpha
        self.decay = decay
        self.num_directions = num_directions  # 采样方向数量
        self.original_weight = original_weight  # 原始梯度方向的权重

        # 确保权重在有效范围内
        assert 0 < self.original_weight < 1, "原始梯度权重必须在(0,1)范围内"

    def get_grad(self, loss, delta, **kwargs):
        """
        计算多样化梯度方向

        1. 获取原始梯度方向
        2. 在原始梯度180度范围内采样多个方向
        3. 对原始梯度赋予较高权重，对采样方向赋予较低权重
        4. 聚合成新的梯度方向
        """
        # 计算原始梯度
        grad_original = torch.autograd.grad(loss, delta, retain_graph=True, create_graph=False)[0]

        # 归一化原始梯度
        grad_norm = torch.norm(grad_original.view(grad_original.shape[0], -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
        grad_normalized = grad_original / (grad_norm + 1e-10)

        # 初始化聚合梯度为加权的原始梯度
        grad_aggregated = self.original_weight * grad_normalized

        # 在原始梯度180度范围内采样多个方向
        remaining_weight = 1.0 - self.original_weight  # 分配给采样方向的总权重
        direction_weight = remaining_weight / self.num_directions  # 每个采样方向的权重

        for _ in tqdm(range(self.num_directions)):
            # 生成随机扰动向量
            perturbation = torch.randn_like(grad_normalized)

            # 确保扰动向量在原始梯度的180度范围内（点积>0表示角度<90度）
            dot_product = torch.sum(perturbation * grad_normalized, dim=[1,2,3], keepdim=True)
            # 如果点积为负（角度>90度），则翻转方向
            perturbation = torch.where(dot_product < 0, -perturbation, perturbation)

            # 归一化扰动向量
            pert_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
            perturbation_normalized = perturbation / (pert_norm + 1e-10)

            # 将归一化的扰动向量加入聚合梯度，并赋予较小的权重
            grad_aggregated += direction_weight * perturbation_normalized

        # 返回聚合后的梯度
        return grad_aggregated