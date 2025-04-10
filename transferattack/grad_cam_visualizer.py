import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class FeatureVisualizer:
    """使用GradCAM等方法可视化模型关注区域
    
    提供多种可视化方法：
    1. GradCAM: 使用梯度加权的特征图
    2. ScoreCAM: 不使用梯度，通过前向传播评估区域贡献
    3. GradCAM++: GradCAM的改进版本
    4. EigenCAM: 使用主成分分析的特征图
    
    可用于分析模型的注意力区域以及对抗样本的效果
    """
    
    def __init__(self, model, target_layers=None, use_cuda=True, method='gradcam'):
        """
        初始化特征可视化器
        
        参数:
            model: PyTorch模型
            target_layers: 目标特征层列表
            use_cuda: 是否使用CUDA
            method: 可视化方法 ('gradcam', 'scorecam', 'gradcam++', 'eigencam')
        """
        self.model = model
        self.device = next(model.parameters()).device
        
        # 如果没有提供目标层，尝试自动查找
        if target_layers is None:
            target_layers = self._find_suitable_layers(model)
            
        if not target_layers:
            raise ValueError("无法找到适合的目标层，请手动指定")
        
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # 选择可视化方法
        if method.lower() == 'gradcam':
            self.cam = GradCAM(model=model, target_layers=target_layers)
        elif method.lower() == 'scorecam':
            self.cam = ScoreCAM(model=model, target_layers=target_layers, )
        elif method.lower() in ['gradcam++', 'gradcampp']:
            self.cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        elif method.lower() == 'eigencam':
            self.cam = EigenCAM(model=model, target_layers=target_layers)
        else:
            print(f"警告: 未知的可视化方法 '{method}', 默认使用 GradCAM")
            self.cam = GradCAM(model=model, target_layers=target_layers)
    
    def _find_suitable_layers(self, model):
        """自动查找模型中适合的层"""
        import torch.nn as nn
        
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
        
        # 优先选择靠近输出的层
        if found_layers:
            # 取倒数第3层或最后一层（如果层数少于3）
            idx = max(0, len(found_layers) - 3)
            return [found_layers[idx][1]]
        else:
            return []
    
    def visualize(self, input_tensor, target_category=None, alpha=0.5):
        """
        生成热力图可视化
        
        参数:
            input_tensor: 输入图像张量 [1, C, H, W]
            target_category: 目标类别索引，如果为None则使用预测最高的类别
            alpha: 热力图与原图混合的透明度
            
        返回:
            可视化结果图像（RGB格式）
        """
        # 确保输入在CPU上并且是numpy格式
        input_tensor = input_tensor.detach().cpu()
        
        # 转换为可显示的图像格式
        img = input_tensor[0].permute(1, 2, 0).numpy()
        
        # 标准化到0-1范围
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # 如果没有指定目标类别，则使用预测的类别
        if target_category is None:
            with torch.no_grad():
                output = self.model(input_tensor.to(self.device))
                target_category = torch.argmax(output).item()
                
        targets = [ClassifierOutputTarget(target_category)]
        
        # 生成热力图
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # 移除批次维度
        
        # 将热力图与原图融合
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True, image_weight=1-alpha)
        
        return visualization, grayscale_cam
        
    def compare_original_and_adversarial(self, original_img, adv_img, target_category=None, alpha=0.6):
        """比较原图和对抗样本的可视化结果"""
        # 生成原图的热力图
        orig_vis, orig_cam = self.visualize(original_img, target_category, alpha)
        
        # 使用对抗样本进行预测
        with torch.no_grad():
            adv_output = self.model(adv_img.to(self.device))
            adv_category = torch.argmax(adv_output).item()
            
        # 使用对抗样本的预测类别生成热力图
        adv_vis, adv_cam = self.visualize(adv_img, adv_category, alpha)
        
        # 创建原图和对抗样本的显示图像
        orig_img_display = (original_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        adv_img_display = (adv_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # 计算对抗扰动（放大以便可视化）
        perturbation = ((adv_img - original_img)[0].permute(1, 2, 0).cpu().numpy() * 10 + 0.5)
        perturbation = np.clip(perturbation, 0, 1)
        
        # 绘制比较图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(orig_img_display)
        plt.title("原始图像")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(adv_img_display)
        plt.title(f"对抗样本 (预测类别: {adv_category})")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(perturbation)
        plt.title("对抗扰动 (10x放大)")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(orig_vis)
        plt.title(f"原图热力图 (类别: {target_category})")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(adv_vis)
        plt.title(f"对抗样本热力图 (类别: {adv_category})")
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        diff_cam = np.abs(adv_cam - orig_cam)
        plt.imshow(diff_cam, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("热力图差异")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 返回可视化结果和热力图
        return {
            'orig_vis': orig_vis,
            'adv_vis': adv_vis,
            'orig_cam': orig_cam,
            'adv_cam': adv_cam,
            'perturbation': perturbation,
            'diff_cam': diff_cam
        }