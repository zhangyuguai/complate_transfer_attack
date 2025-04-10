import torch
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

class FeatureExtractor:
    """使用pytorch_grad_cam库从模型中提取特征图
    
    这个类封装了特征图提取的功能，支持不同的层级并提供干净的访问接口
    """
    def __init__(self, model, target_layers=None, use_cuda=True):
        """
        初始化特征提取器
        
        参数:
            model: PyTorch模型
            target_layers: 需要提取特征的层列表，如果为None则自动搜索合适的层
            use_cuda: 是否使用CUDA
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # 如果没有提供目标层，尝试自动查找
        if target_layers is None:
            target_layers = self._find_suitable_layers(model)
            
        if not target_layers:
            raise ValueError("无法找到适合的目标层，请手动指定")
            
        self.target_layers = target_layers
        self.activations_and_grads = ActivationsAndGradients(
            model, target_layers, None)
            
        self.layer_names = {layer: f"layer_{i}" for i, layer in enumerate(target_layers)}
        self.extracted_features = {}
        self.best_layer = None
        
    def _find_suitable_layers(self, model, candidates=None):
        """自动查找模型中适合的层
        
        查找卷积层或特征层，优先返回靠近输出的层
        """
        import torch.nn as nn
        
        # 默认候选层类型
        if candidates is None:
            candidates = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]
            
        found_layers = []
        
        # 递归查找所有层
        def search_layers(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # 检查是否是候选层类型
                if any(isinstance(child, cand) for cand in candidates):
                    found_layers.append((full_name, child))
                    
                # 递归搜索子层
                search_layers(child, full_name)
                
        search_layers(model)
        
        # 优先选择中间层（既不是最浅也不是最深的层）
        if len(found_layers) >= 3:
            # 选择三分之一和三分之二位置的层
            idx1 = len(found_layers) // 3
            idx2 = 2 * len(found_layers) // 3
            return [found_layers[idx1][1], found_layers[idx2][1]]
        elif found_layers:
            # 如果层数少于3，返回所有找到的层
            return [layer[1] for layer in found_layers]
        else:
            return []
            
    def extract_features(self, x, return_layer=None):
        """提取特征图
        
        参数:
            x: 输入张量
            return_layer: 指定返回特定层的特征，如果为None则返回所有层
            
        返回:
            如果指定return_layer，返回该层的特征张量
            否则返回包含所有层特征的字典
        """
        self.extracted_features = {}
        
        # 使用激活和梯度捕获器运行模型
        _ = self.activations_and_grads(x)
        
        # 获取激活（特征图）
        for i, layer in enumerate(self.target_layers):
            layer_name = self.layer_names[layer]
            activation = self.activations_and_grads.activations[i].detach()
            self.extracted_features[layer_name] = activation
            
        # 如果指定了返回特定层
        if return_layer is not None:
            if isinstance(return_layer, int) and 0 <= return_layer < len(self.target_layers):
                layer = self.target_layers[return_layer]
                layer_name = self.layer_names[layer]
                return self.extracted_features[layer_name]
            elif return_layer in self.extracted_features:
                return self.extracted_features[return_layer]
            else:
                print(f"警告: 未找到指定层 {return_layer}，返回第一个可用层")
                first_key = next(iter(self.extracted_features))
                return self.extracted_features[first_key]
        
        return self.extracted_features
        
    def get_feature_size(self, x):
        """获取各层特征图尺寸"""
        _ = self.extract_features(x)
        return {name: feat.shape for name, feat in self.extracted_features.items()}
        
    def find_best_layer(self, orig_x, adv_x, method='l2'):
        """找到对抗样本和原始图像特征差异最大的层
        
        参数:
            orig_x: 原始图像
            adv_x: 对抗样本
            method: 距离计算方法，'l2'或'cosine'
            
        返回:
            最佳层名称
        """
        orig_features = self.extract_features(orig_x)
        adv_features = self.extract_features(adv_x)
        
        best_score = -float('inf')
        best_layer = None
        
        for name in orig_features.keys():
            orig_feat = orig_features[name]
            adv_feat = adv_features[name]
            
            # 确保特征形状一致
            if orig_feat.shape != adv_feat.shape:
                continue
                
            # 将特征展平为向量
            orig_flat = orig_feat.view(orig_feat.size(0), -1)
            adv_flat = adv_feat.view(adv_feat.size(0), -1)
            
            if method == 'l2':
                # 计算L2距离
                distance = torch.norm(orig_flat - adv_flat, dim=1).mean().item()
            elif method == 'cosine':
                # 计算余弦相似度
                orig_norm = torch.nn.functional.normalize(orig_flat, p=2, dim=1)
                adv_norm = torch.nn.functional.normalize(adv_flat, p=2, dim=1)
                sim = torch.sum(orig_norm * adv_norm, dim=1).mean().item()
                distance = 1 - sim  # 转换为距离
            else:
                raise ValueError(f"不支持的距离计算方法: {method}")
                
            print(f"层 {name}: 特征距离 = {distance:.6f}")
            
            if distance > best_score:
                best_score = distance
                best_layer = name
                
        self.best_layer = best_layer
        print(f"选择最佳特征层: {best_layer}，距离分数: {best_score:.6f}")
        return best_layer
        
    def get_best_layer_features(self, x):
        """从最佳层获取特征"""
        if self.best_layer is None:
            print("警告: 未指定最佳层，使用第一个可用层")
            return self.extract_features(x, return_layer=0)
        else:
            return self.extract_features(x, return_layer=self.best_layer)
            
    def visualize_feature_maps(self, x, layer_name=None, max_maps=16):
        """可视化特征图（用于调试）"""
        import matplotlib.pyplot as plt
        
        features = self.extract_features(x)
        if layer_name is None:
            if self.best_layer is not None:
                layer_name = self.best_layer
            else:
                layer_name = next(iter(features))
                
        if layer_name not in features:
            print(f"警告: 未找到层 {layer_name}，使用第一个可用层")
            layer_name = next(iter(features))
            
        feature_maps = features[layer_name].cpu().numpy()
        batch_size, channels, height, width = feature_maps.shape
        
        # 限制显示的特征图数量
        channels_to_plot = min(channels, max_maps)
        
        plt.figure(figsize=(15, 15))
        for i in range(channels_to_plot):
            plt.subplot(4, channels_to_plot//4 + 1, i+1)
            plt.imshow(feature_maps[0, i], cmap='viridis')
            plt.axis('off')
            plt.title(f'Map {i}')
            
        plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()