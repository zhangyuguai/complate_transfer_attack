import torch

from ..utils import *
from ..gradient.mifgsm import MIFGSM

mid_output = None
mid_grad = None

class FIAWithVarianceStabilization(MIFGSM):
    """
    Feature Importance-aware Attack with Variance Stabilization
    
    不使用动量，仅通过特征方差来稳定梯度更新或调整损失函数
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10,
                 targeted=False, random_start=False, feature_layer='Mixed_5b',
                 norm='linfty', loss='crossentropy', device=None, attack='FIAVariance',
                 drop_rate=0.2, num_ens=30, var_weight=0.1,
                 stabilize_mode='loss', **kwargs):
        # 注意：我们移除了decay参数，因为不使用动量
        super().__init__(model_name, epsilon, alpha, epoch, 0.0, targeted, random_start, norm, loss, device, attack)
        self.num_ens = num_ens
        self.feature_layer = self.find_layer(feature_layer)
        self.drop_rate = drop_rate
        self.var_weight = var_weight
        # 方差稳定化模式: 'loss' 或 'step'
        # - 'loss'：将方差加入损失函数
        # - 'step'：根据方差调整步长
        self.stabilize_mode = stabilize_mode

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
        global mid_output
        mid_output = o

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

    def compute_feature_variance(self, current_features, prev_features):
        """计算当前特征与前一迭代特征之间的方差"""
        if prev_features is None:
            return torch.tensor(0.0).to(self.device)
            
        # 计算特征差异的平方和平均值
        feature_diff = current_features - prev_features
        # 可以考虑不同的方差计算方式：
        # 1. 全局方差 (所有维度)
        variance = feature_diff.pow(2).mean()
        # 2. 按通道计算方差，然后平均
        # variance = feature_diff.pow(2).mean(dim=(2, 3)).mean()
        return variance
       
    def forward(self, data, label, **kwargs):
        """主攻击流程，不使用动量"""
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化对抗扰动
        delta = self.init_delta(data)

        # 注册钩子
        h_forward = self.feature_layer.register_forward_hook(self.__forward_hook)
        h_backward = self.feature_layer.register_full_backward_hook(self.__backward_hook)

        # 第一步：聚合特征梯度 (FIA)
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
            agg_grad[batch_i] /= agg_grad[batch_i].norm(2) + 1e-10
            
        h_backward.remove()
        
        # 保存上一次迭代的特征
        prev_features = None
        
        # 主迭代攻击循环
        for iter_i in range(self.epoch):
            # 获取当前特征
            _ = self.model(self.transform(data + delta))
            current_features = mid_output
            
            # 计算特征方差
            feature_variance = self.compute_feature_variance(current_features, prev_features)
            
            # 保存当前特征用于下一轮计算
            prev_features = current_features.detach()
            
            # 计算FIA的特征损失
            fia_loss = (current_features * agg_grad).sum()
            
            if self.stabilize_mode == 'loss':
                # 方法1：将方差直接加入损失函数作为正则化项
                # 这里使用负号，是为了最小化方差（使特征更稳定）
                final_loss = fia_loss - self.var_weight * feature_variance
                
                # 计算对delta的梯度
                self.model.zero_grad()
                grad = torch.autograd.grad(final_loss, delta, retain_graph=False, create_graph=False)[0]
                
                # 不使用动量，直接更新对抗扰动
                delta = self.update_delta(delta, data, -grad, self.alpha)
                
            else:  # 'step'模式
                # 方法2：根据方差调整步长
                # 计算原始梯度
                self.model.zero_grad()
                grad = torch.autograd.grad(fia_loss, delta, retain_graph=False, create_graph=False)[0]
                
                # 标准化梯度
                norm_grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-10)
                
                # 根据方差调整步长：
                # 如果方差小，说明特征稳定，可以用更大步长
                # 如果方差大，说明特征不稳定，应该用更小步长
                # 这里使用一个简单的反比例关系
                variance_factor = 1.0 / (1.0 + self.var_weight * feature_variance)
                adaptive_alpha = self.alpha * variance_factor
                
                # 直接用调整后的步长更新
                delta = self.update_delta(delta, data, -norm_grad, adaptive_alpha)
        
        # 清理钩子
        h_forward.remove()
        
        return delta.detach()