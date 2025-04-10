from random import random

import torch
import numpy as np
from mpmath import rand

from ..utils import *
from ..attack import Attack

class PerImageHistoricalAttack(Attack):
    """
    基于图片级历史优化的梯度聚合攻击

    对每张图片单独维护历史最佳样本记录，实现更精细的历史信息利用
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, num_neighbors=10,
                 neighbor_radius=0.5, history_size=5, initial_weight=0.4, best_weight=0.6,
                 epoch=10, decay=1.0, targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, **kwargs):
        """
        初始化攻击方法
        """
        super().__init__("PerImageHistoricalAttack", model_name, epsilon,
                         targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_neighbors = num_neighbors
        self.neighbor_radius = neighbor_radius * epsilon
        self.history_size = history_size
        self.initial_weight = initial_weight
        self.best_weight = best_weight

        # 图片级历史记录 - 使用字典存储，键为图片ID
        self.history = {}
        self.prev_delta = None  # 上一次迭代的delta

    def get_image_id(self, image):
        """
        生成图片的唯一标识符
        注意：在实际使用中，如果有真实的图片ID应使用真实ID
        这里使用图片哈希值作为近似ID
        """
        # 使用图片内容的哈希值作为ID
        return hash(image.detach().cpu().flatten().numpy().tobytes())

    def score_adversarial(self, data, delta, label):
        """
        评分函数：评估对抗样本的有效性
        """
        with torch.no_grad():
            x_adv = data + delta
            logits = self.get_logits(x_adv)

            # 计算预测概率
            probs = torch.softmax(logits, dim=1)

            # 非目标攻击：真实类别概率越低越好
            true_probs = probs.gather(1, label.unsqueeze(1))
            # 转换为[0,1]区间，0表示原始类别概率为1（不成功），1表示原始类别概率为0（完全成功）
            scores = 1 - true_probs.squeeze()

            # 确保输出是向量
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

            return scores

    def generate_neighbor_samples(self, data, delta, label, num_samples):
        """
        在当前delta的邻域内生成候选样本
        """
        batch_size = data.shape[0]

        # 准备好存放每个图片的候选样本
        all_candidates = [[] for _ in range(batch_size)]
        all_scores = [[] for _ in range(batch_size)]

        # 对批次中的每张图片处理
        for sample_idx in range(batch_size):
            # 获取单个样本
            single_data = data[sample_idx:sample_idx+1]
            single_delta = delta[sample_idx:sample_idx+1]
            single_label = label[sample_idx:sample_idx+1]

            # 确定中心delta
            if self.prev_delta is not None and self.prev_delta.shape[0] > sample_idx:
                center_delta = self.prev_delta[sample_idx:sample_idx+1].clone()
            else:
                center_delta = single_delta.clone()

            # 生成候选样本
            for _ in range(num_samples):
                # 生成邻域噪声
                noise = torch.zeros_like(single_delta).uniform_(-self.neighbor_radius, self.neighbor_radius).to(self.device)
                # 计算新的delta
                neighbor_delta = center_delta + noise

                #对梯度进行一个随机的小范围冲刺,范围为(0,0.5)
                neighbor_delta *= torch.rand(1).item() * 0.5 + 1



                # 确保在epsilon范围内
                neighbor_delta = torch.clamp(neighbor_delta, -self.epsilon, self.epsilon)

                # 限制在有效图像范围内
                neighbor_delta = clamp(neighbor_delta, img_min-single_data, img_max-single_data)

                # 评分
                score = self.score_adversarial(single_data, neighbor_delta, single_label)

                all_candidates[sample_idx].append(neighbor_delta)
                all_scores[sample_idx].append(score.item())  # 转为标量存储

        # 为每个样本找到最佳候选
        best_deltas = []
        best_scores = []
        best_indices = []

        for sample_idx in range(batch_size):
            if len(all_scores[sample_idx]) > 0:
                best_idx = np.argmax(all_scores[sample_idx])
                best_indices.append(best_idx)
                best_deltas.append(all_candidates[sample_idx][best_idx])
                best_scores.append(all_scores[sample_idx][best_idx])
            else:
                # 极少数情况下可能没有有效候选
                best_indices.append(0)
                best_deltas.append(delta[sample_idx:sample_idx+1])
                best_scores.append(0.0)

        # 将所有最佳delta拼接成一个批次
        best_deltas = torch.cat(best_deltas, dim=0)
        best_scores = torch.tensor(best_scores, device=self.device)

        return best_deltas, best_scores

    def update_history(self, data, deltas, scores):
        """
        更新每张图片的历史最佳样本记录

        参数:
            data: 原始图像批次
            deltas: 当前批次的对抗扰动
            scores: 对应的分数
        """
        batch_size = data.shape[0]

        for i in range(batch_size):
            # 获取图片ID
            img_id = self.get_image_id(data[i])

            # 获取当前样本的delta和分数
            current_delta = deltas[i:i+1]
            current_score = scores[i].item()

            # 如果是新图片，初始化其历史记录
            if img_id not in self.history:
                self.history[img_id] = {
                    'deltas': [],
                    'scores': []
                }

            # 添加新记录
            self.history[img_id]['deltas'].append(current_delta.clone().detach())
            self.history[img_id]['scores'].append(current_score)

            # 如果历史记录超过限制，移除最旧的
            if len(self.history[img_id]['deltas']) > self.history_size:
                self.history[img_id]['deltas'].pop(0)
                self.history[img_id]['scores'].pop(0)

    def aggregate_from_history(self, data, current_delta, initial_delta, label):
        """
        从历史最佳样本和初始样本聚合梯度，每张图片单独处理

        参数:
            data: 原始图像批次
            current_delta: 当前扰动
            initial_delta: 本次迭代初始扰动
            label: 标签

        返回:
            聚合后的梯度
        """
        batch_size = data.shape[0]

        # 计算当前梯度
        current_delta_copy = current_delta.clone().detach().requires_grad_(True)
        logits_current = self.get_logits(data + current_delta_copy)
        loss_current = self.get_loss(logits_current, label)
        current_grad = torch.autograd.grad(loss_current, current_delta_copy)[0]

        # 计算初始梯度
        initial_delta_copy = initial_delta.clone().detach().requires_grad_(True)
        logits_initial = self.get_logits(data + initial_delta_copy)
        loss_initial = self.get_loss(logits_initial, label)
        initial_grad = torch.autograd.grad(loss_initial, initial_delta_copy)[0]

        # 为每个样本准备聚合梯度
        aggregated_grad = torch.zeros_like(current_grad)

        for i in range(batch_size):
            # 获取图片ID
            img_id = self.get_image_id(data[i])

            # 如果图片有历史记录
            if img_id in self.history and len(self.history[img_id]['deltas']) > 0:
                # 获取单张图片的历史数据
                hist_deltas = self.history[img_id]['deltas']
                hist_scores = self.history[img_id]['scores']

                # 计算历史梯度
                hist_grads = []
                for hist_delta in hist_deltas:
                    # 确保形状匹配
                    if hist_delta.shape[0] == 1:
                        hist_delta_copy = hist_delta.clone().detach().requires_grad_(True)

                        # 前向传播
                        single_data = data[i:i+1]
                        single_label = label[i:i+1]

                        logits_hist = self.get_logits(single_data + hist_delta_copy)
                        loss_hist = self.get_loss(logits_hist, single_label)

                        # 计算梯度
                        try:
                            grad_hist = torch.autograd.grad(loss_hist, hist_delta_copy)[0]
                            hist_grads.append(grad_hist)
                        except Exception as e:
                            print(f"计算历史梯度出错: {e}")

                # 如果有有效历史梯度
                if len(hist_grads) > 0:
                    # 计算历史梯度的加权平均
                    weighted_hist_grad = torch.zeros_like(hist_grads[0])

                    # 计算权重 - 越新的样本权重越高
                    weights = []
                    for idx, score in enumerate(hist_scores):
                        recency_weight = (idx + 1) / len(hist_scores)
                        weights.append(recency_weight * score)

                    # 归一化权重
                    weights = np.array(weights)
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)
                    else:
                        weights = np.ones_like(weights) / len(weights)

                    # 加权汇总
                    for idx, grad in enumerate(hist_grads):
                        weighted_hist_grad += weights[idx] * grad

                    # 检查梯度方向冲突
                    current_single_grad = current_grad[i:i+1]
                    initial_single_grad = initial_grad[i:i+1]

                    # 计算梯度点积来检测冲突
                    hist_conflict = self.check_gradient_conflict(current_single_grad, weighted_hist_grad)
                    initial_conflict = self.check_gradient_conflict(current_single_grad, initial_single_grad)

                    # 处理冲突梯度
                    if hist_conflict:
                        weighted_hist_grad = self.project_gradient(weighted_hist_grad, current_single_grad)

                    if initial_conflict:
                        initial_single_grad = self.project_gradient(initial_single_grad, current_single_grad)

                    # 聚合该样本的梯度
                    aggregated_grad[i:i+1] = (1 - self.initial_weight - self.best_weight) * current_single_grad + \
                                             self.initial_weight * initial_single_grad + \
                                             self.best_weight * weighted_hist_grad
                else:
                    # 没有历史梯度，使用当前梯度和初始梯度
                    aggregated_grad[i:i+1] = (1 - self.initial_weight) * current_grad[i:i+1] + \
                                             self.initial_weight * initial_grad[i:i+1]
            else:
                # 没有历史记录，使用当前梯度和初始梯度
                aggregated_grad[i:i+1] = (1 - self.initial_weight) * current_grad[i:i+1] + \
                                         self.initial_weight * initial_grad[i:i+1]

        return aggregated_grad

    def check_gradient_conflict(self, grad1, grad2):
        """
        检查两个梯度是否存在方向冲突

        参数:
            grad1, grad2: 两个梯度

        返回:
            布尔值，True表示存在冲突
        """
        # 将梯度展平
        g1_flat = grad1.view(grad1.shape[0], -1)
        g2_flat = grad2.view(grad2.shape[0], -1)

        # 计算余弦相似度
        similarity = torch.cosine_similarity(g1_flat, g2_flat, dim=1)

        # 如果余弦相似度为负，说明梯度有冲突
        return (similarity < 0).item()

    def project_gradient(self, grad_to_project, reference_grad):
        """
        将梯度投影到参考梯度方向上

        参数:
            grad_to_project: 需要投影的梯度
            reference_grad: 参考梯度

        返回:
            投影后的梯度
        """
        # 将梯度展平
        g_proj_flat = grad_to_project.view(grad_to_project.shape[0], -1)
        g_ref_flat = reference_grad.view(reference_grad.shape[0], -1)

        # 计算投影系数
        dot_product = torch.sum(g_proj_flat * g_ref_flat, dim=1, keepdim=True)
        ref_norm_squared = torch.sum(g_ref_flat * g_ref_flat, dim=1, keepdim=True)

        # 避免除以零
        safe_denominator = torch.clamp(ref_norm_squared, min=1e-10)
        projection_coef = dot_product / safe_denominator

        # 投影
        projected_flat = projection_coef * g_ref_flat

        # 恢复原始形状
        projected_grad = projected_flat.view_as(grad_to_project)

        return projected_grad

    def forward(self, data, label, **kwargs):
        """
        执行攻击
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

            # 生成并评估邻域样本
            best_deltas, best_scores = self.generate_neighbor_samples(
                data, delta, label, self.num_neighbors
            )

            # 更新每张图片的历史记录
            self.update_history(data, best_deltas, best_scores)

            # 聚合梯度
            aggregated_grad = self.aggregate_from_history(
                data, best_deltas, initial_delta, label
            )

            # 更新动量
            momentum = self.get_momentum(aggregated_grad, momentum)

            # 更新扰动
            delta = self.update_delta(delta, data, momentum, self.alpha)

            # 保存当前delta供下一轮使用
            self.prev_delta = delta.clone().detach()

            # 打印进度
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