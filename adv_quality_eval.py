import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import pandas as pd
import seaborn as sns

# 尝试导入LPIPS，如果不存在，提供安装说明
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available. Install with: pip install lpips")

class AdversarialQualityEvaluator:
    """
    对抗样本质量评估器
    
    支持以下指标：
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - MSE (Mean Squared Error)
    - L1距离 (平均绝对误差)
    - L∞距离 (最大扰动)
    - 扰动可视化
    
    参数:
        device (str): 计算设备 ('cuda' 或 'cpu')
        use_lpips (bool): 是否使用LPIPS度量（需要额外安装）
    """
    
    def __init__(self, device='cuda', use_lpips=True):
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.use_lpips = use_lpips and LPIPS_AVAILABLE
        
        # 初始化LPIPS模型
        if self.use_lpips:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
    
    def _ensure_tensor(self, img):
        """确保输入是PyTorch张量，并且在正确的设备上"""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            
            # 如果是单通道图像，扩展为3通道
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1)  # HWC -> CHW
                
            # 如果值范围是[0, 255]，则归一化到[0, 1]
            if img.max() > 1.0:
                img = img / 255.0
        
        if img.dim() == 3:  # 添加批次维度
            img = img.unsqueeze(0)
            
        return img.to(self.device)
    
    def compute_mse(self, original, adversarial):
        """计算均方误差 (MSE)"""
        original = self._ensure_tensor(original)
        adversarial = self._ensure_tensor(adversarial)
        
        mse = F.mse_loss(original, adversarial, reduction='none')
        return mse.mean([1, 2, 3]).cpu().numpy()
    
    def compute_l1(self, original, adversarial):
        """计算L1距离 (平均绝对误差)"""
        original = self._ensure_tensor(original)
        adversarial = self._ensure_tensor(adversarial)
        
        l1 = F.l1_loss(original, adversarial, reduction='none')
        return l1.mean([1, 2, 3]).cpu().numpy()
    
    def compute_linf(self, original, adversarial):
        """计算L∞距离 (最大绝对误差)"""
        original = self._ensure_tensor(original)
        adversarial = self._ensure_tensor(adversarial)
        
        linf = (original - adversarial).abs().flatten(1)
        return linf.max(dim=1)[0].cpu().numpy()
    
    def compute_psnr(self, original, adversarial):
        """计算峰值信噪比 (PSNR)"""
        original = self._ensure_tensor(original)
        adversarial = self._ensure_tensor(adversarial)
        
        # 转换为NumPy进行计算
        original_np = original.permute(0, 2, 3, 1).cpu().numpy()
        adversarial_np = adversarial.permute(0, 2, 3, 1).cpu().numpy()
        
        psnr_values = []
        for i in range(original_np.shape[0]):
            psnr_values.append(psnr(original_np[i], adversarial_np[i], data_range=1.0))
            
        return np.array(psnr_values)
    
    def compute_ssim(self, original, adversarial):
        """计算结构相似性指数 (SSIM)"""
        original = self._ensure_tensor(original)
        adversarial = self._ensure_tensor(adversarial)
        
        # 转换为NumPy进行计算
        original_np = original.permute(0, 2, 3, 1).cpu().numpy()
        adversarial_np = adversarial.permute(0, 2, 3, 1).cpu().numpy()
        
        ssim_values = []
        for i in range(original_np.shape[0]):
            # 多通道SSIM
            ssim_value = 0
            for c in range(original_np.shape[3]):
                ssim_value += ssim(original_np[i, :, :, c], 
                                 adversarial_np[i, :, :, c],
                                 data_range=1.0)
            ssim_values.append(ssim_value / original_np.shape[3])
            
        return np.array(ssim_values)
    
    def compute_lpips(self, original, adversarial):
        """计算LPIPS分数 (感知相似度)"""
        if not self.use_lpips:
            return np.array([-1.0] * len(original))
            
        original = self._ensure_tensor(original)
        adversarial = self._ensure_tensor(adversarial)
        
        with torch.no_grad():
            lpips_values = self.lpips_model(original, adversarial)
            
        return lpips_values.squeeze().cpu().numpy()
    
    def evaluate(self, original, adversarial):
        """评估所有图像质量指标"""
        metrics = {
            'mse': self.compute_mse(original, adversarial),
            'l1': self.compute_l1(original, adversarial),
            'linf': self.compute_linf(original, adversarial),
            'psnr': self.compute_psnr(original, adversarial),
            'ssim': self.compute_ssim(original, adversarial)
        }
        
        if self.use_lpips:
            metrics['lpips'] = self.compute_lpips(original, adversarial)
            
        return metrics
    
    def summarize_metrics(self, metrics):
        """汇总指标，计算平均值和标准差"""
        summary = {}
        for metric_name, values in metrics.items():
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
        
        return summary
    
    def visualize_perturbation(self, original, adversarial, idx=0, save_path=None, 
                               figsize=(15, 5), cmap='viridis', enhanced_contrast=10.0):
        """
        可视化对抗扰动
        
        参数:
            original: 原始图像
            adversarial: 对抗样本
            idx: 批次中的图像索引
            save_path: 保存图像的路径 (如果指定)
            figsize: 图像大小
            cmap: 热力图颜色映射
            enhanced_contrast: 扩大扰动对比度的系数
        """
        original = self._ensure_tensor(original)
        adversarial = self._ensure_tensor(adversarial)
        
        # 提取单张图像
        orig_img = original[idx].cpu().permute(1, 2, 0).numpy()
        adv_img = adversarial[idx].cpu().permute(1, 2, 0).numpy()
        
        # 计算扰动
        perturbation = adv_img - orig_img
        
        # 增强对比度以便于可视化
        enhanced_perturbation = perturbation * enhanced_contrast
        enhanced_perturbation = np.clip(enhanced_perturbation + 0.5, 0, 1)
        
        # 计算相关指标
        batch_metrics = self.evaluate(original[idx:idx+1], adversarial[idx:idx+1])
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 原始图像
        axes[0].imshow(np.clip(orig_img, 0, 1))
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 对抗样本
        axes[1].imshow(np.clip(adv_img, 0, 1))
        axes[1].set_title('对抗样本')
        axes[1].axis('off')
        
        # 扰动可视化
        perturbation_vis = axes[2].imshow(enhanced_perturbation, cmap=cmap)
        axes[2].set_title('扰动 (对比度增强)')
        axes[2].axis('off')
        fig.colorbar(perturbation_vis, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 添加指标信息
        metric_text = f"PSNR: {batch_metrics['psnr'][0]:.2f} dB\n"
        metric_text += f"SSIM: {batch_metrics['ssim'][0]:.4f}\n"
        metric_text += f"L∞距离: {batch_metrics['linf'][0]:.4f}\n"
        
        if self.use_lpips:
            metric_text += f"LPIPS: {batch_metrics['lpips'][0]:.4f}"
        
        plt.figtext(0.5, 0.01, metric_text, ha='center', fontsize=12, 
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def batch_evaluate_save(self, original_dir, adversarial_dir, output_dir,
                           csv_filename='quality_metrics.csv', 
                           visualize_samples=5, img_extension='.png'):
        """
        批量评估文件夹中的图像并保存结果
        
        参数:
            original_dir: 原始图像目录
            adversarial_dir: 对抗样本目录
            output_dir: 输出目录
            csv_filename: CSV结果文件名
            visualize_samples: 随机可视化的样本数量
            img_extension: 图像文件扩展名
        """
        os.makedirs(output_dir, exist_ok=True)
        #@print(f"评估原始图像: {original_dir} 和对抗样本: {adversarial_dir}...")
        # 获取所有图像文件
        image_files = [f for f in os.listdir(original_dir) if f.endswith(img_extension)]
        
        # 随机选择样本进行可视化
        if visualize_samples > 0:
            vis_indices = np.random.choice(len(image_files), 
                                     min(visualize_samples, len(image_files)), 
                                     replace=False)
        else:
            vis_indices = []
        
        all_metrics = []
        print(image_files)
        for i, img_file in enumerate(tqdm(image_files, desc="评估图像质量")):
            try:
                # 加载图像
                orig_path = os.path.join(original_dir, img_file)
                adv_path = os.path.join(adversarial_dir, img_file)
                
                if not os.path.exists(orig_path):
                    print(f"警告: 原始图像 {img_file} 不存在，跳过。")
                    continue
                    
                if not os.path.exists(adv_path):
                    print(f"警告: 对抗样本 {img_file} 不存在，跳过。")
                    continue
                
                try:
                    orig_img = Image.open(orig_path).convert('RGB')
                    adv_img = Image.open(adv_path).convert('RGB')
                except Exception as img_err:
                    print(f"无法打开图像 {img_file}: {str(img_err)}")
                    continue
                
                transform = transforms.ToTensor()
                orig_tensor = transform(orig_img)
                adv_tensor = transform(adv_img)
                
                # 计算指标
                metrics = self.evaluate(orig_tensor, adv_tensor)
                
                # 保存每个图像的指标
                img_metrics = {metric: values[0] for metric, values in metrics.items()}
                img_metrics['image'] = img_file
                all_metrics.append(img_metrics)
                
                # 可视化
                if i in vis_indices:
                    vis_path = os.path.join(output_dir, f'visual_{img_file}')
                    self.visualize_perturbation(orig_tensor, adv_tensor, 
                                              save_path=vis_path)
            
            except Exception as e:
                print(f"处理 {img_file} 时出错: {str(e)}")
        
        # 保存CSV结果
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            csv_path = os.path.join(output_dir, csv_filename)
            metrics_df.to_csv(csv_path, index=False)
            
            # 生成摘要统计和可视化
            self._generate_summary_report(metrics_df, output_dir)
        
        return all_metrics
    
    def _generate_summary_report(self, metrics_df, output_dir):
        """生成质量指标摘要报告和可视化"""
        # 计算摘要统计
        summary = metrics_df.describe()
        summary_path = os.path.join(output_dir, 'summary_statistics.csv')
        summary.to_csv(summary_path)
        
        # 创建箱线图可视化
        plt.figure(figsize=(12, 8))
        
        metrics_to_plot = [col for col in metrics_df.columns if col != 'image']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 3, i+1)
            sns.boxplot(y=metrics_df[metric])
            plt.title(f'{metric.upper()} 分布')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'metrics_distribution.png')
        plt.savefig(plot_path, dpi=300)
        
        # 创建相关性热力图
        plt.figure(figsize=(10, 8))
        corr = metrics_df[metrics_to_plot].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('质量指标相关性')
        plt.tight_layout()
        corr_path = os.path.join(output_dir, 'metrics_correlation.png')
        plt.savefig(corr_path, dpi=300)

    def compute_lpips(self, original, adversarial):
        """计算LPIPS分数 (感知相似度)，添加了更健壮的异常处理"""
        if not self.use_lpips:
            return np.array([-1.0] * original.shape[0])
            
        try:
            original = self._ensure_tensor(original)
            adversarial = self._ensure_tensor(adversarial)
            
            # 确保图像尺寸在LPIPS能接受的范围内
            # LPIPS偏好尺寸为64的倍数的图像
            with torch.no_grad():
                lpips_values = []
                for i in range(original.shape[0]):
                    try:
                        # 每次处理一个图像对
                        orig_img = original[i:i+1]
                        adv_img = adversarial[i:i+1]
                        
                        # 计算LPIPS距离
                        dist = self.lpips_model(orig_img, adv_img)
                        
                        # 确保结果是标量
                        if isinstance(dist, torch.Tensor):
                            if dist.ndim > 0:
                                dist = dist.mean()
                            lpips_values.append(dist.item())
                        else:
                            lpips_values.append(float(dist))
                            
                    except Exception as e:
                        print(f"LPIPS计算单个图像时出错: {str(e)}")
                        lpips_values.append(-1.0)  # 错误时使用-1标记
                        
                return np.array(lpips_values)
                
        except Exception as e:
            print(f"LPIPS整体计算出错: {str(e)}")
            return np.array([-1.0] * original.shape[0])


def evaluate_with_basename_matching(evaluator, orig_dir, adv_dir, output_dir):
    """基于文件基础名进行匹配的图像质量评估，增强尺寸处理"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取原始目录和对抗样本目录中的所有文件
    orig_files = os.listdir(orig_dir)
    adv_files = os.listdir(adv_dir)
    
    # 创建基础名到文件名的映射
    orig_basename_map = {}
    for f in orig_files:
        basename = os.path.splitext(f)[0]
        orig_basename_map[basename] = f
        
    adv_basename_map = {}
    for f in adv_files:
        basename = os.path.splitext(f)[0]
        adv_basename_map[basename] = f
        
    # 找到共有的基础名
    common_basenames = set(orig_basename_map.keys()) & set(adv_basename_map.keys())
    print(f"找到{len(common_basenames)}个匹配的图像对")
    
    if not common_basenames:
        print("警告: 未找到匹配的图像对！")
        print(f"原始图像目录中的前5个基础名: {list(orig_basename_map.keys())[:5]}")
        print(f"对抗样本目录中的前5个基础名: {list(adv_basename_map.keys())[:5]}")
        return []
        
    # 随机选择样本进行可视化
    visualize_samples = 5
    vis_basenames = np.random.choice(list(common_basenames), 
                               min(visualize_samples, len(common_basenames)), 
                               replace=False)
    
    all_metrics = []
    for i, basename in enumerate(tqdm(common_basenames, desc="评估图像质量")):
        try:
            # 获取完整文件名
            orig_file = orig_basename_map[basename]
            adv_file = adv_basename_map[basename]
            
            # 构建完整路径
            orig_path = os.path.join(orig_dir, orig_file)
            adv_path = os.path.join(adv_dir, adv_file)
            
            # 加载图像
            orig_img = Image.open(orig_path).convert('RGB')
            adv_img = Image.open(adv_path).convert('RGB')
            
            # 检查图像尺寸
            orig_size = orig_img.size
            adv_size = adv_img.size
            
            # 调整对抗样本的尺寸，但限制最大尺寸以提高性能
            if orig_size != adv_size:
                # 如果原始图像太大，先进行降采样
                max_size = 1024  # 设置最大尺寸限制
                if orig_size[0] > max_size or orig_size[1] > max_size:
                    # 计算缩放比例，保持纵横比
                    ratio = min(max_size / orig_size[0], max_size / orig_size[1])
                    new_size = (int(orig_size[0] * ratio), int(orig_size[1] * ratio))
                    print(f"原始图像过大，调整 {basename} 的尺寸: {orig_size} -> {new_size}")
                    orig_img = orig_img.resize(new_size, Image.BICUBIC)
                    adv_img = adv_img.resize(new_size, Image.BICUBIC)
                else:
                    print(f"调整对抗样本 {basename} 的尺寸: {adv_size} -> {orig_size}")
                    adv_img = adv_img.resize(orig_size, Image.BICUBIC)
            
            # 转换为张量
            transform = transforms.ToTensor()
            orig_tensor = transform(orig_img)
            adv_tensor = transform(adv_img)
            
            # 安全地分别计算每个指标
            try:
                mse = evaluator.compute_mse(orig_tensor, adv_tensor)
                l1 = evaluator.compute_l1(orig_tensor, adv_tensor)
                linf = evaluator.compute_linf(orig_tensor, adv_tensor)
                psnr_val = evaluator.compute_psnr(orig_tensor, adv_tensor)
                ssim_val = evaluator.compute_ssim(orig_tensor, adv_tensor)
                
                # 初始化指标字典
                metrics = {
                    'mse': mse,
                    'l1': l1,
                    'linf': linf,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                }
                
                # 单独尝试LPIPS
                try:
                    if evaluator.use_lpips:
                        lpips_val = evaluator.compute_lpips(orig_tensor, adv_tensor)
                        metrics['lpips'] = lpips_val
                except Exception as lpips_err:
                    print(f"计算 {basename} 的LPIPS失败: {str(lpips_err)}")
                    metrics['lpips'] = np.array([-1.0])  # 失败时使用-1标记
                
            except Exception as metric_err:
                print(f"计算 {basename} 的质量指标失败: {str(metric_err)}")
                continue
            
            # 保存每个图像的指标
            img_metrics = {metric: values[0] for metric, values in metrics.items()}
            img_metrics['image'] = basename
            all_metrics.append(img_metrics)
            
            # 可视化
            if basename in vis_basenames:
                try:
                    vis_path = os.path.join(output_dir, f'visual_{basename}.png')
                    evaluator.visualize_perturbation(orig_tensor, adv_tensor, 
                                                  save_path=vis_path)
                except Exception as vis_err:
                    print(f"可视化 {basename} 失败: {str(vis_err)}")
        
        except Exception as e:
            print(f"处理 {basename} 时出错: {str(e)}")
    
    # 保存CSV结果
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(output_dir, 'quality_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        
        # 生成摘要统计和可视化
        try:
            evaluator._generate_summary_report(metrics_df, output_dir)
        except Exception as report_err:
            print(f"生成摘要报告失败: {str(report_err)}")
    
    return all_metrics
# 使用示例
# 将main函数部分修改为:

if __name__ == "__main__":
    # 创建评估器
    evaluator = AdversarialQualityEvaluator(device='cuda', use_lpips=True)
    
    original_dir = r"E:\python_workspace\TransferAttack-main\data\data\images"
    adversarial_dir = r"E:\python_workspace\TransferAttack-main\shuffle_crop_mirror_10"
    output_dir = r"E:\python_workspace\TransferAttack-main\output"
    
    # 检查路径是否存在
    print(f"原始图像目录存在: {os.path.exists(original_dir)}")
    print(f"对抗样本目录存在: {os.path.exists(adversarial_dir)}")
    
    # 实现基于文件基础名的评估
    def evaluate_with_basename_matching(evaluator, orig_dir, adv_dir, output_dir):
        """基于文件基础名进行匹配的图像质量评估，并处理尺寸不匹配问题"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取原始目录和对抗样本目录中的所有文件
        orig_files = os.listdir(orig_dir)
        adv_files = os.listdir(adv_dir)
        
        # 创建基础名到文件名的映射
        orig_basename_map = {}
        for f in orig_files:
            basename = os.path.splitext(f)[0]
            orig_basename_map[basename] = f
            
        adv_basename_map = {}
        for f in adv_files:
            basename = os.path.splitext(f)[0]
            adv_basename_map[basename] = f
            
        # 找到共有的基础名
        common_basenames = set(orig_basename_map.keys()) & set(adv_basename_map.keys())
        print(f"找到{len(common_basenames)}个匹配的图像对")
        
        if not common_basenames:
            print("警告: 未找到匹配的图像对！")
            print(f"原始图像目录中的前5个基础名: {list(orig_basename_map.keys())[:5]}")
            print(f"对抗样本目录中的前5个基础名: {list(adv_basename_map.keys())[:5]}")
            return []
            
        # 随机选择样本进行可视化
        visualize_samples = 5
        vis_basenames = np.random.choice(list(common_basenames), 
                                min(visualize_samples, len(common_basenames)), 
                                replace=False)
        
        all_metrics = []
        for i, basename in enumerate(tqdm(common_basenames, desc="评估图像质量")):
            try:
                # 获取完整文件名
                orig_file = orig_basename_map[basename]
                adv_file = adv_basename_map[basename]
                
                # 构建完整路径
                orig_path = os.path.join(orig_dir, orig_file)
                adv_path = os.path.join(adv_dir, adv_file)
                
                # 加载图像
                orig_img = Image.open(orig_path).convert('RGB')
                adv_img = Image.open(adv_path).convert('RGB')
                
                # 确保对抗样本与原始图像具有相同的尺寸
                if orig_img.size != adv_img.size:
                    print(f"调整 {basename} 的尺寸: {adv_img.size} -> {orig_img.size}")
                    adv_img = adv_img.resize(orig_img.size, Image.BICUBIC)
                
                # 转换为张量
                transform = transforms.ToTensor()
                orig_tensor = transform(orig_img)
                adv_tensor = transform(adv_img)
                
                # 计算指标
                metrics = evaluator.evaluate(orig_tensor, adv_tensor)
                
                # 保存每个图像的指标
                img_metrics = {metric: values[0] for metric, values in metrics.items()}
                img_metrics['image'] = basename
                all_metrics.append(img_metrics)
                
                # 可视化
                if basename in vis_basenames:
                    vis_path = os.path.join(output_dir, f'visual_{basename}.png')
                    evaluator.visualize_perturbation(orig_tensor, adv_tensor, 
                                                save_path=vis_path)
            
            except Exception as e:
                print(f"处理 {basename} 时出错: {str(e)}")
        
        # 保存CSV结果
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            csv_path = os.path.join(output_dir, 'quality_metrics.csv')
            metrics_df.to_csv(csv_path, index=False)
            
            # 生成摘要统计和可视化
            evaluator._generate_summary_report(metrics_df, output_dir)
        
        return all_metrics
    
    # 执行评估
    if os.path.exists(original_dir) and os.path.exists(adversarial_dir):
        # 检查目录内容
        orig_files = os.listdir(original_dir)
        adv_files = os.listdir(adversarial_dir)
        
        print(f"原始图像目录中的文件数: {len(orig_files)}")
        print(f"对抗样本目录中的文件数: {len(adv_files)}")
        
        # 显示文件扩展名情况
        orig_extensions = set(os.path.splitext(f)[1].upper() for f in orig_files)
        adv_extensions = set(os.path.splitext(f)[1].upper() for f in adv_files)
        
        print(f"原始图像目录中的扩展名: {orig_extensions}")
        print(f"对抗样本目录中的扩展名: {adv_extensions}")
        
        # 执行基于基础名匹配的评估
        evaluate_with_basename_matching(evaluator, original_dir, adversarial_dir, output_dir)
    else:
        print("目录不存在，请检查路径")