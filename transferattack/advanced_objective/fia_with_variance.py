import torch
import numpy as np

from ..utils import *
from ..gradient.mifgsm import MIFGSM

# 全局变量用于钩子函数
mid_output = None
mid_grad = None
all_features = []

class ImprovedFIANoMomentum(MIFGSM):
    """
    Improved FIA Attack with Multi-Sample Feature Variance, without momentum mechanism
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=0., num_ens=30,
                 targeted=False, random_start=False, feature_layer='layer2',
                 norm='linfty', loss='crossentropy', device=None, attack='ImprovedFIA',
                 drop_rate=0.3, num_sample=20, radius=7, var_weight=0.5, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, 0.0, targeted, random_start, norm, loss, device, attack)
        self.num_ens = num_ens
        self.feature_layer = self.find_layer(feature_layer)
        self.drop_rate = drop_rate
        self.radius = radius
        self.num_sample = num_sample
        self.var_weight = var_weight

    def find_layer(self, layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        for layer in parser:
            if layer not in m._modules.keys():
                print("Selected layer is not in Model")
                exit()
            else:
                m = m._modules.get(layer)
        return m

    def __forward_hook(self, m, i, o):
        global mid_output, all_features
        mid_output = o
        all_features.append(o.clone().detach())

    def __backward_hook(self, m, i, o):
        global mid_grad
        mid_grad = o

    def drop(self, data):
        x_drop = torch.zeros(data.size()).cuda()
        x_drop.copy_(data).detach()
        x_drop.requires_grad = True
        Mask = torch.bernoulli(torch.ones_like(x_drop) * (1 - self.drop_rate))
        x_drop = x_drop * Mask
        return x_drop

    def get_samples(self, x, grad_direction):
        """使用多种策略生成多个样本"""
        samples = []
        # 线性方向采样
        factors = np.linspace(-self.radius, self.radius, num=self.num_sample)
        for factor in factors:
            samples.append(x + factor * self.alpha * grad_direction)
        return torch.cat(samples)

    def compute_feature_variance(self, features):
        """计算特征方差，用于正则化项"""
        if len(features) <= 1:
            return torch.tensor(0.0).to(self.device)

        # 计算特征均值
        mean_feature = sum(features) / len(features)

        # 计算方差
        variance = 0
        for feat in features:
            variance += (feat - mean_feature).pow(2)
        variance = variance / len(features)

        # 返回总体方差
        return variance.mean()

    def forward(self, data, label, **kwargs):
        """改进的攻击流程，移除动量机制"""
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化对抗扰动
        delta = self.init_delta(data)
        delta.requires_grad = True
        # 注册钩子
        h_forward = self.feature_layer.register_forward_hook(self.__forward_hook)
        h_backward = self.feature_layer.register_full_backward_hook(self.__backward_hook)

        # 第一步：聚合特征梯度 (标准FIA方式)
        global all_features
        agg_grad = 0
        for _ in range(self.num_ens):
            x_drop = self.drop(data)
            output_random = self.model(x_drop)
            output_random = torch.softmax(output_random, 1)
            loss = 0
            for batch_i in range(data.shape[0]):
                loss += output_random[batch_i][label[batch_i]]
            self.model.zero_grad()
            loss.backward()
            agg_grad += mid_grad[0].detach()

        # 归一化聚合特征梯度
        for batch_i in range(data.shape[0]):
            agg_grad[batch_i] /= (agg_grad[batch_i].norm(2) + 1e-10)

        # 第一阶段完成，移除反向钩子
        h_backward.remove()

        # 主攻击循环 (无动量)
        for i in range(self.epoch):
            all_features = []  # 重置特征收集列表

            # 先获取当前特征
            _ = self.model(self.transform(data + delta))
            current_feature = mid_output.detach()

            # 计算当前梯度方向
            loss_current = (current_feature * agg_grad).sum()
            self.model.zero_grad()
            grad_current = torch.autograd.grad(loss_current, delta, retain_graph=False, create_graph=False)[0]

            # 归一化梯度方向 (用于采样，但不直接用于更新)
            grad_norm = grad_current / (grad_current.abs().mean(dim=(1,2,3), keepdim=True) + 1e-10)

            # 根据当前梯度方向生成多个样本
            samples = self.get_samples(data + delta, grad_norm)

            # 前向传播所有样本
            _ = self.model(samples)

            # 此时 all_features 包含了所有样本的特征输出

            # 计算特征方差作为正则项
            feature_variance = self.compute_feature_variance(all_features)

            # 计算多样本特征损失
            feature_losses = []
            for feat in all_features:
                feature_losses.append((feat * agg_grad).sum())

            # 合并损失 (特征方差作为正则项)
            # 负号表示我们希望减小方差以增加稳定性
            total_loss = sum(feature_losses) - self.var_weight * feature_variance

            # 计算最终梯度
            self.model.zero_grad()
            final_grad = torch.autograd.grad(total_loss, delta, retain_graph=False, create_graph=False)[0]

            # 归一化最终梯度 (替代动量)
            normalized_grad = final_grad / (final_grad.abs().mean(dim=(1,2,3), keepdim=True) + 1e-10)

            # 直接使用归一化梯度更新，不用动量
            delta = self.update_delta(delta, data, -normalized_grad, self.alpha)

        # 清理钩子
        h_forward.remove()

        return delta.detach()