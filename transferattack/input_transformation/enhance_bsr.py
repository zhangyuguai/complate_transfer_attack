import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from typing import List, Tuple
from ..utils import *
from ..gradient.mifgsm import MIFGSM

class GradAwareTransform:
    """基于梯度显著性的动态块变换操作"""
    def __init__(
            self,
            base_blocks: int = 3,
            max_blocks: int = 6,
            rotation_degree: int = 30,
            min_block_size: int = 32
    ):
        self.base_blocks = base_blocks
        self.max_blocks = max_blocks
        self.rotation = T.RandomRotation(degrees=(-rotation_degree, rotation_degree))
        self.min_block_size = min_block_size  # 最小分块尺寸

    def __call__(self, x: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        """
        执行显著性引导的变换操作
        Args:
            x: 输入图像张量 (B,C,H,W)
            saliency_map: 显著性图 (B,1,H,W)
        Returns:
            变换后的张量 (B,C,H,W)
        """
        B, C, H, W = x.shape

        # 动态分块策略
        h_splits, w_splits = self.dynamic_partition(saliency_map)

        processed = []
        for b in range(B):
            # 纵向分块处理
            h_blocks = self.process_axis(
                img=x[b],
                splits=h_splits[b],
                axis=2,
                saliency=saliency_map[b].sum(dim=(1,2))
            )

            # 横向分块处理
            rotated_blocks = []
            for h_blk in h_blocks:
                w_blocks = self.process_axis(
                    img=h_blk,
                    splits=w_splits[b],
                    axis=3,
                    saliency=saliency_map[b].sum(dim=(0,2))
                )
                rotated_blocks.append(torch.cat(w_blocks, dim=2))

            processed.append(torch.cat(rotated_blocks, dim=1))

        return torch.stack(processed)

    def dynamic_partition(self, saliency: torch.Tensor) -> Tuple[List[List[int]]]:
        """生成动态分块方案"""
        B = saliency.size(0)
        h_splits, w_splits = [], []

        for b in range(B):
            # 显著性投影分析
            h_proj = saliency[b].sum(dim=(1,2)).cpu().numpy()
            w_proj = saliency[b].sum(dim=(0,2)).cpu().numpy()

            # 自适应分块数
            dynamic_blocks = min(
                self.max_blocks,
                max(self.base_blocks, int(np.log(h_proj.max() + 1e-8)))
            )

            h_splits.append(self.adaptive_split(h_proj, dynamic_blocks, saliency.shape[2]))
            w_splits.append(self.adaptive_split(w_proj, dynamic_blocks, saliency.shape[3]))

        return h_splits, w_splits

    def adaptive_split(self, projection: np.ndarray, num_blocks: int, max_length: int) -> List[int]:
        """基于显著性投影的智能分块算法"""
        # 平滑处理
        smoothed = np.convolve(projection, np.ones(3)/3, mode='same')

        # 寻找显著边界
        boundaries = []
        grad = np.abs(np.gradient(smoothed))
        threshold = np.percentile(grad, 75)

        for i in range(1, len(grad)-1):
            if grad[i] > threshold and grad[i] > grad[i-1] and grad[i] > grad[i+1]:
                boundaries.append(i)

        # 动态调整分块
        if len(boundaries) >= num_blocks - 1:
            selected = sorted(np.random.choice(boundaries, num_blocks-1, replace=False))
        else:
            selected = np.linspace(0, max_length, num_blocks+1)[1:-1].astype(int).tolist()

        # 生成分块方案
        splits = np.diff([0] + sorted(selected) + [max_length])
        return self.apply_min_size(splits)

    def apply_min_size(self, splits: np.ndarray) -> List[int]:
        """确保最小分块尺寸"""
        adjusted = []
        remaining = 0
        for s in splits:
            if s + remaining < self.min_block_size:
                remaining += s
            else:
                adjusted.append(s + remaining)
                remaining = 0
        if remaining > 0:
            adjusted[-1] += remaining
        return [int(s) for s in adjusted if s > 0]

    def process_axis(self, img: torch.Tensor, splits: List[int], axis: int, saliency: torch.Tensor) -> List[torch.Tensor]:
        """单轴处理流程"""
        blocks = torch.split(img, splits, dim=axis)

        # 显著性排序洗牌
        block_importance = [saliency[..., :b.shape[axis]].mean() for b in blocks]
        order = np.argsort(block_importance)[::-1]  # 按显著性降序排列

        # 块处理
        processed = []
        for idx in order:
            block = blocks[idx]
            # 添加随机旋转
            rotated = self.rotation(block)
            processed.append(rotated)

        return processed