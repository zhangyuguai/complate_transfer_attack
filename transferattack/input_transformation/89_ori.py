import torch
import random
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class BSR(MIFGSM):
    """
    简化版BSR (Block Shuffle)对抗攻击
    
    继承自MIFGSM攻击方法，只保留块混洗功能，移除旋转变换
    特点：
    1. 只进行块混洗，不进行旋转
    2. 将裁剪后的样本作为样本增强，单独计算损失
    3. 裁剪区域与分块区域保持一致
    
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
        num_block=2,  # 默认分为3块
        targeted=False, 
        random_start=False, 
        norm='linfty', 
        loss='crossentropy',
        device=None, 
        attack='BSR', 
        scale_range=(0.5, 0.8),  # 裁剪缩放范围
        crop_prob=1,  # 块裁剪概率
        crop_loss_weight=0.7,  # 裁剪样本损失权重
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
        self.scale_range = scale_range      # 随机裁剪的比例范围
        self.crop_prob = crop_prob          # 块裁剪概率
        self.crop_loss_weight = crop_loss_weight  # 裁剪样本损失权重
        self.di_prob = di_prob              # 多样性输入概率
        
        print(f"初始化只打乱版 {attack} 攻击: 分块数={self.num_block}, 混洗副本数={num_scale}")
        print(f"裁剪缩放范围={scale_range}, 块裁剪概率={crop_prob}, 裁剪样本损失权重={crop_loss_weight}")
        print(f"使用损失函数: 交叉熵(CrossEntropy)损失")
        print(f"注意：已禁用旋转功能，只保留块打乱")
        
        # 用于保存当前迭代中的随机选择，确保分块和裁剪一致
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
        """沿着指定维度处理分块（只分块打乱，不旋转），返回两个版本：只打乱和打乱+裁剪"""
        dim_size = x.size(dim)
        if dim_size < self.num_block:
            # 只返回原始版本和裁剪版本，不进行旋转
            cropped = self.crop_block(x)
            return [x], [cropped]
            
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
        
        # 对打乱后的块进行裁剪
        cropped_blocks = []
        for block in shuffled_blocks:
            cropped_block = self.crop_block(block)
            cropped_blocks.append(cropped_block)
        
        return shuffled_blocks, cropped_blocks

    def shuffle(self, x, batch_id):
        """只进行分块打乱，不旋转，返回两个版本：只打乱的和打乱裁剪的"""
        try:
            # 高度和宽度维度
            dims = [2, 3]
            random.shuffle(dims)
            
            if x.size(dims[0]) <= 1 or x.size(dims[1]) <= 1:
                cropped = self.crop_block(x)
                return x, cropped
            
            # 第一维度上的分块处理
            key_id = f"batch_{batch_id}"
            shuffled_blocks, cropped_blocks = self.process_blocks_along_dim(x, dims[0], key_id)
            
            if len(shuffled_blocks) == 1:
                return shuffled_blocks[0], cropped_blocks[0]
            
            # 第二维度上的分块处理
            final_shuffled_blocks = []
            final_cropped_blocks = []
            
            for i, (shuffle_block, cropped_block) in enumerate(zip(shuffled_blocks, cropped_blocks)):
                if shuffle_block.size(dims[1]) <= 1:
                    final_shuffled_blocks.append(shuffle_block)
                    final_cropped_blocks.append(cropped_block)
                else:
                    sub_key_id = f"{key_id}_sub_{i}"
                    inner_shuffled, inner_cropped = self.process_blocks_along_dim(shuffle_block, dims[1], sub_key_id)
                    
                    if len(inner_shuffled) > 1:
                        final_shuffled_blocks.append(torch.cat(inner_shuffled, dim=dims[1]))
                        final_cropped_blocks.append(torch.cat(inner_cropped, dim=dims[1]))
                    else:
                        final_shuffled_blocks.append(inner_shuffled[0])
                        final_cropped_blocks.append(inner_cropped[0])
            
            return torch.cat(final_shuffled_blocks, dim=dims[0]), torch.cat(final_cropped_blocks, dim=dims[0])
            
        except Exception as e:
            print(f"混洗操作出错: {e}")
            cropped = self.crop_block(x)
            return x, cropped

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
        """BSR变换：返回打乱版本和打乱裁剪版本，不进行旋转"""
        x = x.to(self.device)
        shuffled_results = []
        cropped_results = []
        
        # 每次迭代开始时清空块索引缓存
        self.current_block_indices = {}
        
        for i in range(self.num_scale):
            try:
                # 先应用多样性输入变换
                # x_di = self.diverse_input(x)
                x_di = x
                # 块打乱和裁剪，不进行旋转
                shuffled, cropped = self.shuffle(x_di, batch_id=i)
                
                if shuffled is not None and shuffled.numel() > 0:
                    shuffled_results.append(shuffled)
                if cropped is not None and cropped.numel() > 0:
                    cropped_results.append(cropped)
                    
            except Exception as e:
                print(f"变换过程出错: {e}")
                pass
        
        # 处理空结果情况
        if not shuffled_results:
            shuffled_results = [x for _ in range(self.num_scale)]
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
        cropped_results = ensure_consistent_size(cropped_results, target_size)
        
        # 分别拼接两种结果
        shuffled_batch = torch.cat(shuffled_results, dim=0)
        cropped_batch = torch.cat(cropped_results, dim=0)
        
        return shuffled_batch, cropped_batch

    def get_loss(self, logits, label):
        """使用交叉熵损失计算分类损失"""
        if self.targeted:
            return -self.loss(logits, label)
        else:
            return self.loss(logits, label)
    def gradient_smoothing(self, grad, kernel_size=3):
        """应用高斯平滑到梯度"""
        # 沿着通道应用平滑
        channels = grad.size(1)
        weight = torch.ones(channels, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        weight = weight.to(grad.device)
        
        # 使用填充保持尺寸
        padding = kernel_size // 2
        smoothed_grad = F.conv2d(grad, weight, groups=channels, padding=padding)
        
        return smoothed_grad
    def forward(self, data, label, **kwargs):
        """执行只打乱版BSR攻击（分别计算打乱版本和裁剪版本的损失）"""
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
        #定义shuffle_weight
        self.shuffle_weight = 1
        # 迭代攻击
        for i in tqdm(range(self.epoch), desc=f"{self.attack} on {self.model_name} crop*********************"):
            try:
                # 对抗样本 = 原图 + 扰动
                x_adv = data + delta
                
                # 执行BSR变换，获取打乱版本和裁剪版本
                x_shuffled, x_cropped = self.transform(x_adv)
                
                # 前向传播 - 打乱版本
                logits_shuffled = self.model(x_shuffled)
                
                # 前向传播 - 裁剪版本
                logits_cropped = self.model(x_cropped)
                
                # 计算标签副本
                repeated_label = label.repeat(self.num_scale)
                
                # 计算两种版本的损失
                loss_shuffled = self.get_loss(logits_shuffled, repeated_label)
                loss_cropped = self.get_loss(logits_cropped, repeated_label)
                
                # 组合损失 - 注意裁剪版本损失的权重
                loss = self.shuffle_weight*loss_shuffled + self.crop_loss_weight * loss_cropped
                
                # 定期打印损失
                if i % 5 == 0:
                    print(f"迭代 {i}: 打乱版本损失={loss_shuffled.item():.4f}, " 
                          f"裁剪版本损失={loss_cropped.item():.4f}, 总损失={loss.item():.4f}")
                
                # 计算梯度
                grad = self.get_grad(loss, delta)
                #应用随机梯度平滑
                # 随机应用梯度平滑
                if random.random() < 0.7:  # 70%概率应用
                    grad = self.gradient_smoothing(grad)
                
                # 随机添加梯度噪声
                if random.random() < 0.3:  # 30%概率添加噪声
                    noise_strength = random.uniform(0.05, 0.15)
                    grad = grad + torch.randn_like(grad) * noise_strength * torch.norm(grad)
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