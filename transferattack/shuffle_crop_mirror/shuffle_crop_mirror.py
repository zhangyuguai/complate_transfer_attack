import torch
import random
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from torchvision.models import resnet50
import cv2
import numpy as np
from torchvision.models import resnet50, resnet101, vgg19

from .dct import dct_2d, idct_2d
from ..utils import *
from ..gradient.mifgsm import MIFGSM



class BSR(MIFGSM):
    """
    组合损失BSR (Block Shuffle with separate Mirroring and Cropping losses)对抗攻击
    
    继承自MIFGSM攻击方法，结合镜像和裁剪两种增强的损失
    特点：
    1. 只进行块混洗，不进行旋转
    2. 单独计算镜像版本和裁剪版本的损失，并结合起来
    3. 支持水平镜像、垂直镜像和双向镜像
    
    参数:
        model_name (str): 代理模型名称
        epsilon (float): 扰动预算
        alpha (float): 每步扰动大小
        epoch (int): 迭代次数
        decay (float): 动量衰减系数
        num_scale (int): 混洗副本数量
        num_block (int): 图像分块数量
        targeted (bool): 是否为目标攻击
        random_start (bool): 是否随机初始化扰动
        norm (str): 扰动范数类型，l2/linfty
        loss (str): 损失函数类型，默认为'crossentropy'
        mirror_prob (float): 块镜像概率
        mirror_type (str): 镜像类型，'horizontal'/'vertical'/'both'/'random'
        mirror_loss_weight (float): 镜像样本损失权重
        scale_range (tuple): 随机裁剪的比例范围
        crop_prob (float): 块裁剪概率
        crop_loss_weight (float): 裁剪样本损失权重
        di_prob (float): 多样性输入概率
    """
    
    def __init__(
        self, 
        model_name, 
        epsilon=16/255, 
        alpha=1.6/255, 
        epoch=10, 
        decay=1.0, 
        num_scale=10, 
        num_block=2,  # 默认分为2块
        targeted=False, 
        random_start=False, 
        norm='linfty', 
        loss='crossentropy',
        device=None, 
        attack='BSRCombo', 
        mirror_prob=1.0,  # 块镜像概率
        mirror_type='both',  # 镜像类型：水平/垂直/双向/随机
        mirror_loss_weight=0.6,  # 镜像样本损失权重
        scale_range=(0.4, 0.8),  # 裁剪缩放范围
        crop_prob=1.0,  # 块裁剪概率
        crop_loss_weight=0.6,  # 裁剪样本损失权重
        di_prob=0.5,  # 多样性输入概率
        **kwargs
    ):
        # 初始化MIFGSM基类
        super().__init__(
            model_name=model_name, 
            epsilon=epsilon, 
            alpha=alpha, 
            epoch=epoch, 
            decay=decay, 
            targeted=targeted, 
            random_start=random_start, 
            norm=norm, 
            loss=loss, 
            device=device, 
            attack=attack
        )
        
        # BSR特有参数
        self.num_scale = num_scale
        self.num_block = min(num_block, 5)  # 安全限制块数
        self.mirror_prob = mirror_prob      # 块镜像概率
        self.mirror_type = mirror_type      # 镜像类型
        self.mirror_loss_weight = mirror_loss_weight  # 镜像样本损失权重
        self.scale_range = scale_range      # 随机裁剪的比例范围
        self.crop_prob = crop_prob          # 块裁剪概率
        self.crop_loss_weight = crop_loss_weight  # 裁剪样本损失权重
        self.di_prob = di_prob              # 多样性输入概率
        
        print(f"初始化组合损失版 {attack} 攻击: 分块数={self.num_block}, 混洗副本数={num_scale}")
        print(f"镜像类型={mirror_type}, 块镜像概率={mirror_prob}, 镜像样本损失权重={mirror_loss_weight}")
        print(f"裁剪缩放范围={scale_range}, 块裁剪概率={crop_prob}, 裁剪样本损失权重={crop_loss_weight}")
        print(f"使用损失函数: 交叉熵(CrossEntropy)损失")
        print(f"注意：已禁用旋转功能，分别计算镜像和裁剪损失")
        
        # 用于保存当前迭代中的随机选择，确保分块和镜像一致
        self.current_block_indices = {}
    
    def get_length(self, length):
        """生成随机分块长度，确保每个块至少有一定的最小长度"""
        min_size = max(1, length // (self.num_block * 2))
        max_blocks = length // min_size
        actual_blocks = min(self.num_block, max_blocks)
        
        if actual_blocks <= 1:
            return (length,)
            
        rand = torch.rand(actual_blocks, device=self.device)
        remaining_length = length - (min_size * actual_blocks)
        extra_lengths = (rand / rand.sum() * remaining_length).round().int()
        block_lengths = torch.full((actual_blocks,), min_size, device=self.device) + extra_lengths
        block_lengths[-1] += (length - block_lengths.sum())
        
        return tuple(block_lengths.tolist())
    def mirror_block(self, x_block):
        """对单个块进行镜像操作"""
        batch_size, c, h, w = x_block.shape
        
        if h <= 4 or w <= 4:  # 如果块太小，直接返回
            return x_block
        
        results = []
        for i in range(batch_size):
            img = x_block[i:i+1]
            
            # 只有当概率满足时才进行镜像
            if random.random() < self.mirror_prob:
                # 确定镜像类型
                if self.mirror_type == 'random':
                    mirror_choice = random.choice(['horizontal', 'vertical', 'both'])
                else:
                    mirror_choice = self.mirror_type
                
                # 应用镜像变换
                if mirror_choice == 'horizontal':
                    img_mirrored = torch.flip(img, dims=[3])  # 水平翻转
                elif mirror_choice == 'vertical':
                    img_mirrored = torch.flip(img, dims=[2])  # 垂直翻转
                elif mirror_choice == 'both':
                    img_mirrored = torch.flip(img, dims=[2, 3])  # 同时水平和垂直翻转
                
                results.append(img_mirrored)
            else:
                results.append(img)
        return torch.cat(results, dim=0)

    def crop_block(self, x_block):
        """对单个块进行裁剪和调整大小"""
        batch_size, c, h, w = x_block.shape
        
        if h <= 4 or w <= 4:  # 如果块太小，直接返回
            return x_block
        
        results = []
        for i in range(batch_size):
            img = x_block[i:i+1]
            
            # 只有当概率满足时才进行裁剪
            if random.random() < self.crop_prob:
                # 随机确定裁剪比例
                scale = random.uniform(self.scale_range[0], self.scale_range[1])
                # 计算裁剪尺寸
                crop_h = max(2, int(h * scale))
                crop_w = max(2, int(w * scale))
                
                # 随机确定裁剪位置
                top = random.randint(0, max(0, h - crop_h))
                left = random.randint(0, max(0, w - crop_w))
                
                # 裁剪
                img_cropped = img[:, :, top:min(top+crop_h, h), left:min(left+crop_w, w)]
                
                # 调整回原始尺寸
                img_resized = F.interpolate(img_cropped, size=(h, w), mode='bilinear', align_corners=False)
                results.append(img_resized)
            else:
                results.append(img)
        
        return torch.cat(results, dim=0)

    def process_blocks_along_dim(self, x, dim, key_id):
        """沿着指定维度处理分块，返回三个版本：只打乱、打乱+镜像、打乱+裁剪"""
        dim_size = x.size(dim)
        if dim_size < self.num_block:
            # 只返回原始版本、镜像版本和裁剪版本
            mirrored = self.mirror_block(x)
            cropped = self.crop_block(x)
            return [x], [mirrored], [cropped]
        # 获取块长度
        lengths = self.get_length(dim_size)
        if any(l <= 0 for l in lengths):
            base_length = dim_size // len(lengths)
            lengths = [base_length] * (len(lengths) - 1)
            lengths.append(dim_size - sum(lengths))
        
        # 分块
        x_blocks = list(x.split(lengths, dim=dim))
        
        # 生成随机顺序或使用缓存的顺序
        cache_key = f"{key_id}_{dim}"
        if cache_key not in self.current_block_indices:
            indices = list(range(len(x_blocks)))
            random.shuffle(indices)
            self.current_block_indices[cache_key] = indices
        else:
            indices = self.current_block_indices[cache_key]
        
        # 随机打乱块顺序
        shuffled_blocks = [x_blocks[i] for i in indices]
        
        # 对打乱后的块分别进行镜像和裁剪
        mirrored_blocks = []
        cropped_blocks = []
        
        for block in shuffled_blocks:
            mirrored_block = self.mirror_block(block)
            mirrored_blocks.append(mirrored_block)
            
            cropped_block = self.crop_block(block)
            cropped_blocks.append(cropped_block)
        
        return shuffled_blocks, mirrored_blocks, cropped_blocks

    def shuffle(self, x, batch_id):
        """分块打乱处理，返回三个版本：只打乱、打乱+镜像、打乱+裁剪"""
        try:
            # 高度和宽度维度
            dims = [2, 3]
            random.shuffle(dims)
            
            if x.size(dims[0]) <= 1 or x.size(dims[1]) <= 1:
                mirrored = self.mirror_block(x)
                cropped = self.crop_block(x)
                return x, mirrored, cropped
            
            # 第一维度上的分块处理
            key_id = f"batch_{batch_id}"
            shuffled_blocks, mirrored_blocks, cropped_blocks = self.process_blocks_along_dim(x, dims[0], key_id)
            #展示一下结果
            # print(f"shuffled_blocks: {shuffled_blocks}")plt.figure(figsize=(12, 6))
            #
            #        # 选择一个随机图像进行展示
            batch_idx = random.randint(0, x.size(0)-1)

            # 展示原始图像
            plt.subplot(1, 3, 1)
            orig_img = x[batch_idx].detach().cpu().permute(1, 2, 0).numpy()
            # 归一化显示
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
            plt.imshow(orig_img)
            plt.title('原始图像')

            # 展示打乱块
            if len(shuffled_blocks) > 0:
                plt.subplot(1, 3, 2)
                shuffled_img = shuffled_blocks[0][batch_idx].detach().cpu().permute(1, 2, 0).numpy()
                shuffled_img = (shuffled_img - shuffled_img.min()) / (shuffled_img.max() - shuffled_img.min() + 1e-8)
                plt.imshow(shuffled_img)
                plt.title('打乱块')

            # 展示镜像块
            if len(mirrored_blocks) > 0:
                plt.subplot(1, 3, 3)
                mirror_img = mirrored_blocks[0][batch_idx].detach().cpu().permute(1, 2, 0).numpy()
                mirror_img = (mirror_img - mirror_img.min()) / (mirror_img.max() - mirror_img.min() + 1e-8)
                plt.imshow(mirror_img)
                plt.title('镜像块')

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            if len(shuffled_blocks) == 1:
                return shuffled_blocks[0], mirrored_blocks[0], cropped_blocks[0]
            
            # 第二维度上的分块处理
            final_shuffled_blocks = []
            final_mirrored_blocks = []
            final_cropped_blocks = []
            
            for i, (shuffle_block, mirror_block, crop_block) in enumerate(zip(shuffled_blocks, mirrored_blocks, cropped_blocks)):
                if shuffle_block.size(dims[1]) <= 1:
                    final_shuffled_blocks.append(shuffle_block)
                    final_mirrored_blocks.append(mirror_block)
                    final_cropped_blocks.append(crop_block)
                else:
                    sub_key_id = f"{key_id}_sub_{i}"
                    inner_shuffled, inner_mirrored, inner_cropped = self.process_blocks_along_dim(shuffle_block, dims[1], sub_key_id)
                    
                    if len(inner_shuffled) > 1:
                        final_shuffled_blocks.append(torch.cat(inner_shuffled, dim=dims[1]))
                        final_mirrored_blocks.append(torch.cat(inner_mirrored, dim=dims[1]))
                        final_cropped_blocks.append(torch.cat(inner_cropped, dim=dims[1]))
                    else:
                        final_shuffled_blocks.append(inner_shuffled[0])
                        final_mirrored_blocks.append(inner_mirrored[0])
                        final_cropped_blocks.append(inner_cropped[0])
            
            return torch.cat(final_shuffled_blocks, dim=dims[0]), torch.cat(final_mirrored_blocks, dim=dims[0]), torch.cat(final_cropped_blocks, dim=dims[0])
            
        except Exception as e:
            print(f"混洗操作出错: {e}")
            mirrored = self.mirror_block(x)
            cropped = self.crop_block(x)
            return x, mirrored, cropped

    def diverse_input(self, x):
        """多样性输入变换 (DI)"""
        if random.random() < self.di_prob:
            batch_size, c, h, w = x.shape
            # 随机填充参数
            pad_h = random.randint(0, max(1, h // 8))
            pad_w = random.randint(0, max(1, w // 8))
            
            # 应用填充
            x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
            
            # 调整回原始尺寸
            x_resized = F.interpolate(x_padded, size=(h, w), mode='bilinear', align_corners=False)
            return x_resized
        return x

    def transform(self, x):
        """BSR变换：返回打乱版本、镜像版本和裁剪版本"""
        x = x.to(self.device)
        shuffled_results = []
        mirrored_results = []
        cropped_results = []
        
        # 每次迭代开始时清空块索引缓存
        self.current_block_indices = {}
        
        for i in range(self.num_scale):
            try:
                # 可以选择是否应用多样性输入变换
                #x_di = self.diverse_input(x) if self.di_prob > 0 else x
                x_di = x
                # 块打乱、镜像和裁剪
                shuffled, mirrored, cropped = self.shuffle(x_di, batch_id=i)
                
                if shuffled is not None and shuffled.numel() > 0:
                    shuffled_results.append(shuffled)
                if mirrored is not None and mirrored.numel() > 0:
                    mirrored_results.append(mirrored)
                if cropped is not None and cropped.numel() > 0:
                    cropped_results.append(cropped)
                    
            except Exception as e:
                print(f"变换过程出错: {e}")
                pass
        
        # 处理空结果情况
        if not shuffled_results:
            shuffled_results = [x for _ in range(self.num_scale)]
        if not mirrored_results:
            mirrored_results = [x for _ in range(self.num_scale)]
        if not cropped_results:
            cropped_results = [x for _ in range(self.num_scale)]
        
        # 确保所有结果尺寸一致
        def ensure_consistent_size(results_list, target_size):
            unified_results = []
            for result in results_list:
                try:
                    if result.shape[-2:] != target_size[-2:]:
                        resized = F.interpolate(result, size=target_size[-2:], mode='bilinear', align_corners=False)
                        unified_results.append(resized)
                    else:
                        unified_results.append(result)
                except Exception as e:
                    print(f"尺寸调整出错: {e}")
                    unified_results.append(F.interpolate(x, size=target_size[-2:], mode='bilinear', align_corners=False))
            return unified_results
        
        target_size = x.shape
        shuffled_results = ensure_consistent_size(shuffled_results, target_size)
        mirrored_results = ensure_consistent_size(mirrored_results, target_size)
        cropped_results = ensure_consistent_size(cropped_results, target_size)
        
        # 分别拼接三种结果
        shuffled_batch = torch.cat(shuffled_results, dim=0)
        mirrored_batch = torch.cat(mirrored_results, dim=0)
        cropped_batch = torch.cat(cropped_results, dim=0)
        
        return shuffled_batch, mirrored_batch, cropped_batch

    def get_loss(self, logits, label):
        """使用交叉熵损失计算分类损失"""
        if self.targeted:
            return -self.loss(logits, label)
        else:
            return self.loss(logits, label)
    
    def forward(self, data, label, **kwargs):
        """执行组合损失版BSR攻击（分别计算打乱版本、镜像版本和裁剪版本的损失）"""
        # 处理目标攻击情况
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        
        # 确保数据在正确设备上
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # 初始化扰动和动量
        delta = self.init_delta(data)
        delta.requires_grad = True
        
        self.momentum = torch.zeros_like(data)
        self.shuffle_weight = 1.0
        
        # 迭代攻击
        for i in tqdm(range(self.epoch), desc=f"{self.attack} on {self.model_name} combo*********************"):
            try:
                # 对抗样本 = 原图 + 扰动
                x_adv = data + delta
                # #加入频域扰动尝试
                # adv_freq =dct_2d(x_adv)
                # logits_freq = self.model(adv_freq)
                # #计算频域损失
                # loss_freq = self.get_loss(logits_freq, label)
                # #计算频域梯度
                # grad_dct = self.get_grad(loss_freq, adv_freq)
                # #转回空间域
                # grad_dct = idct_2d(grad_dct).sign()



                # 执行BSR变换，获取三种版本
                x_shuffled, x_mirrored, x_cropped = self.transform(x_adv)
                
                # 前向传播 - 三种版本
                logits_shuffled = self.model(x_shuffled)
                logits_mirrored = self.model(x_mirrored)
                logits_cropped = self.model(x_cropped)
                
                # 计算标签副本
                repeated_label = label.repeat(self.num_scale)
                
                # 计算三种版本的损失
                loss_shuffled = self.get_loss(logits_shuffled, repeated_label)
                loss_mirrored = self.get_loss(logits_mirrored, repeated_label)
                loss_cropped = self.get_loss(logits_cropped, repeated_label)
                
                # 组合损失 - 使用权重
                loss = (self.shuffle_weight * loss_shuffled + 
                        self.mirror_loss_weight * loss_mirrored + 
                        self.crop_loss_weight * loss_cropped)
                
                # 定期打印损失
                if i % 5 == 0:
                    print(f"迭代 {i}: 打乱版本损失={loss_shuffled.item():.4f}, " 
                          f"镜像版本损失={loss_mirrored.item():.4f}, "
                          f"裁剪版本损失={loss_cropped.item():.4f}, "
                          f"总损失={loss.item():.4f}")



                # 计算梯度
                grad = self.get_grad(loss, delta)
                
                # 使用MIFGSM动量更新
                self.momentum = self.get_momentum(grad, self.momentum)


                # 更新扰动
                with torch.no_grad():
                    delta = self.update_delta(delta, data, self.momentum, self.alpha)
                    delta.requires_grad = True
                    
            except Exception as e:
                print(f"迭代 {i} 发生错误: {e}")
                # 发生错误时继续下一次迭代
            
            # 每次迭代后清空块索引缓存，确保下次迭代生成新的随机块
            self.current_block_indices = {}
        
        # 返回最终扰动
        return delta.detach()
    
