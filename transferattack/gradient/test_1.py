import torch
import numpy as np
from ..utils import *
from ..attack import Attack

class HistoricalNeighborhoodSearchAttack(Attack):
    """
    基于历史优化样本的梯度聚合攻击

    该方法在每次迭代中:
    1. 生成当前的主要对抗样本方向
    2. 在上一次对抗扰动的邻域范围内生成多个候选样本
    3. 为每个候选样本计算评分
    4. 记录当前迭代中最佳样本和初始样本
    5. 根据历史最佳样本和初始样本聚合梯度，指导新的攻击方向
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, num_neighbors=10,
                 neighbor_radius=0.5, history_size=5, initial_weight=0.4, best_weight=0.6,
                 epoch=10, decay=1.0, targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, **kwargs):
        """
        初始化攻击方法

        参数:
            model_name (str): 代理模型名称
            epsilon (float): 扰动预算
            alpha (float): 步长
            num_neighbors (int): 每次迭代生成的邻域样本数量
            neighbor_radius (float): 邻域采样半径，相对于epsilon
            history_size (int): 保存的历史最佳样本数量
            initial_weight (float): 初始样本梯度的权重
            best_weight (float): 最佳样本梯度的权重
            epoch (int): 迭代次数
            decay (float): 动量计算的衰减因子
            targeted (bool): 是否为定向攻击
            random_start (bool): 是否随机初始化delta
            norm (str): 扰动范数类型，'l2'或'linfty'
            loss (str): 损失函数类型
            device (torch.device): 计算设备
        """
        super().__init__("HistoricalNeighborhoodSearchAttack", model_name, epsilon,
                         targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_neighbors = num_neighbors
        self.neighbor_radius = neighbor_radius * epsilon
        self.history_size = history_size
        self.initial_weight = initial_weight
        self.best_weight = best_weight

        # 存储历史信息
        self.history_best_deltas = []  # 历史最佳delta
        self.history_best_scores = []  # 历史最佳分数
        self.prev_delta = None         # 上一次迭代的delta

    def score_adversarial(self, data, delta, label):
        """
        评分函数：评估对抗样本的有效性

        参数:
            data: 原始数据
            delta: 对抗扰动
            label: 真实标签（非目标攻击）或目标标签（目标攻击）

        返回:
            分数值（越高越好）
        """
        with torch.no_grad():
            x_adv = data + delta
            logits = self.get_logits(x_adv)

            # 计算预测概率
            probs = torch.softmax(logits, dim=1)

            if self.targeted:
                # 目标攻击：目标类别概率越高越好
                target_probs = probs.gather(1, label.unsqueeze(1))
                scores = target_probs.squeeze()
            else:
                # 非目标攻击：真实类别概率越低越好
                true_probs = probs.gather(1, label.unsqueeze(1))
                # 转换为[0,1]区间，0表示原始类别概率为1（不成功），1表示原始类别概率为0（完全成功）
                scores = 1 - true_probs.squeeze()

            return scores

    def generate_neighbor_samples(self, data, delta, label, num_samples):
        """
        在当前delta的邻域内生成候选样本

        参数:
            data: 原始数据
            delta: 当前扰动
            label: 标签
            num_samples: 生成样本数量

        返回:
            候选样本列表、分数列表、最佳样本索引
        """
        if self.prev_delta is None:
            # 首次迭代，使用当前delta
            center_delta = delta
        else:
            # 使用上一次的delta作为中心
            center_delta = self.prev_delta.clone()

        # 确保形状匹配
        if center_delta.shape != delta.shape:
            center_delta = delta.clone()

        candidate_deltas = []
        scores = []

        # 生成候选样本
        for _ in range(num_samples):
            # 生成邻域噪声
            noise = torch.zeros_like(delta).uniform_(-self.neighbor_radius, self.neighbor_radius).to(self.device)
            # 计算新的delta
            neighbor_delta = center_delta + noise

            # 确保在epsilon范围内
            if self.norm == 'linfty':
                neighbor_delta = torch.clamp(neighbor_delta, -self.epsilon, self.epsilon)
            else:  # l2范数
                neighbor_delta = neighbor_delta.renorm(p=2, dim=0, maxnorm=self.epsilon)

            # 限制在有效图像范围内
            neighbor_delta = clamp(neighbor_delta, img_min-data, img_max-data)

            # 评分
            score = self.score_adversarial(data, neighbor_delta, label)

            candidate_deltas.append(neighbor_delta)
            scores.append(score)

        # 转换为张量
        scores = torch.stack(scores)

        # 检查scores的维度
        if len(scores.shape) == 1:
            # 如果是一维张量，直接使用这些分数
            avg_scores = scores
        elif len(scores.shape) == 2:
            # 如果是二维张量，对第1维度取平均
            avg_scores = torch.mean(scores, dim=1)
        else:
            # 如果是更高维，对除了第0维以外的所有维度取平均
            avg_scores = torch.mean(scores.view(scores.shape[0], -1), dim=1)

        # 找到最佳样本索引
        best_idx = torch.argmax(avg_scores).item()

        return candidate_deltas, avg_scores, best_idx

    def update_history(self, best_delta, best_score):
        """
        更新历史最佳样本记录
        """
        # 添加新的最佳样本
        self.history_best_deltas.append(best_delta.clone().detach())
        self.history_best_scores.append(best_score.clone().detach())

        # 如果历史记录超出大小限制，移除最旧的记录
        if len(self.history_best_deltas) > self.history_size:
            self.history_best_deltas.pop(0)
            self.history_best_scores.pop(0)

    def aggregate_from_history(self, current_delta, initial_delta, data, label):
        """
        从历史最佳样本和初始样本聚合梯度

        参数:
            current_delta: 当前扰动
            initial_delta: 本次迭代初始扰动
            data: 原始数据
            label: 标签

        返回:
            聚合后的梯度
        """
        # 如果没有历史记录，直接返回当前梯度
        if len(self.history_best_deltas) == 0:
            # 计算当前梯度
            current_delta_copy = current_delta.clone().detach().requires_grad_(True)
            logits = self.get_logits(data + current_delta_copy)
            loss = self.get_loss(logits, label)
            current_grad = torch.autograd.grad(loss, current_delta_copy)[0]
            return current_grad

        # 计算当前delta的梯度
        current_delta_copy = current_delta.clone().detach().requires_grad_(True)
        logits_current = self.get_logits(data + current_delta_copy)
        loss_current = self.get_loss(logits_current, label)
        current_grad = torch.autograd.grad(loss_current, current_delta_copy)[0]

        # 计算本次迭代初始delta的梯度
        initial_delta_copy = initial_delta.clone().detach().requires_grad_(True)
        logits_initial = self.get_logits(data + initial_delta_copy)
        loss_initial = self.get_loss(logits_initial, label)
        initial_grad = torch.autograd.grad(loss_initial, initial_delta_copy)[0]

        # 计算历史最佳样本的梯度
        best_grads = []
        best_weights = []

        for i, hist_delta in enumerate(self.history_best_deltas):
            # 确保形状匹配
            if hist_delta.shape != current_delta.shape:
                continue

            hist_delta_copy = hist_delta.clone().detach().requires_grad_(True)
            logits_hist = self.get_logits(data + hist_delta_copy)
            loss_hist = self.get_loss(logits_hist, label)

            try:
                grad_hist = torch.autograd.grad(loss_hist, hist_delta_copy)[0]
                best_grads.append(grad_hist)

                # 分配权重 - 越新的样本权重越高
                recency_weight = (i + 1) / len(self.history_best_deltas)
                best_weights.append(recency_weight * self.history_best_scores[i])
            except Exception as e:
                print(f"计算历史梯度出错: {e}")
                continue

        # 如果没有有效的历史梯度，返回当前梯度和初始梯度的加权和
        if len(best_grads) == 0:
            return (1 - self.initial_weight) * current_grad + self.initial_weight * initial_grad

        # 归一化权重
        best_weights = torch.stack(best_weights)
        if torch.sum(best_weights) > 0:
            best_weights = best_weights / torch.sum(best_weights)
        else:
            best_weights = torch.ones_like(best_weights) / len(best_weights)

        # 计算加权梯度
        weighted_grad = torch.zeros_like(current_grad)
        for i, grad in enumerate(best_grads):
            weighted_grad += best_weights[i] * grad

        # 最终聚合：当前梯度、初始梯度和历史最佳梯度
        # aggregated_grad = (1 - self.initial_weight - self.best_weight) * current_grad + \
        #                   self.initial_weight * initial_grad + \
        #                   self.best_weight * weighted_grad
        aggregated_grad = (1) * current_grad + \
                          self.best_weight * weighted_grad
        return aggregated_grad

    def forward(self, data, label, **kwargs):
        """
        执行攻击

        参数:
            data: 输入图像
            label: 真实标签（非定向攻击）或目标标签（定向攻击）
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # 目标标签

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化扰动
        delta = self.init_delta(data)
        momentum = 0

        for i in range(self.epoch):
            # 保存本次迭代的初始delta
            initial_delta = delta.clone()

            # 生成邻域候选样本
            candidate_deltas, scores, best_idx = self.generate_neighbor_samples(
                data, delta, label, self.num_neighbors
            )

            # 获取最佳候选样本
            best_delta = candidate_deltas[best_idx]
            best_score = scores[best_idx]

            # 更新历史记录
            self.update_history(best_delta, best_score)

            # 从历史记录和初始样本聚合梯度
            aggregated_grad = self.aggregate_from_history(
                best_delta, initial_delta, data, label
            )

            # 更新动量
            momentum = self.get_momentum(aggregated_grad, momentum)

            # 更新扰动
            delta = self.update_delta(delta, data, momentum, self.alpha)

            # 保存当前delta供下一轮使用
            self.prev_delta = delta.clone().detach()

            # 每隔几轮打印进度
            if (i + 1) % 5 == 0 or i == 0:
                with torch.no_grad():
                    logits = self.get_logits(data + delta)
                    if self.targeted:
                        success_rate = (torch.argmax(logits, dim=1) == label).float().mean().item()
                        print(f"迭代 {i+1}/{self.epoch}, 目标类成功率: {success_rate:.4f}")
                    else:
                        success_rate = (torch.argmax(logits, dim=1) != label).float().mean().item()
                        print(f"迭代 {i+1}/{self.epoch}, 攻击成功率: {success_rate:.4f}")

        return delta.detach()