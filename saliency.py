import torch
import torch.nn.functional as F

def get_saliency_map(model, x, label):
    """
    计算并返回输入图像 x 的显著性区域图 (Saliency Map).
    基于被攻击的模型在给定标签 label 下对图像每个像素点的梯度响应.

    参数:
        model (nn.Module): 用于计算梯度的模型, 应在评估模式 (model.eval()) 下使用.
        x (torch.Tensor): 输入图像张量, 形状通常为 [B, C, H, W].
        label (torch.Tensor): 与 x 对应的标签, 形状为 [B], 元素为类别索引 (int).

    返回:
        torch.Tensor: 显著性图, 形状与 x 相同 (或 [B, H, W] 只保留空间维度).
    """
    # 确保模型在评估模式
    model.eval()

    # 确保可以对输入图像的像素执行梯度运算
    x = x.clone().detach().requires_grad_(True)

    # 前向传播，获得对各类别的预测分值
    logits = model(x)

    # 获取目标类别分数 (若是多类别分类, index_select 或 gather)
    # 假设 label 形状 = [B], 则对每个样本取对应的logit
    # 每个样本只取对应label位置上的分值
    scores = logits.gather(dim=1, index=label.unsqueeze(-1)).squeeze()

    # 反向传播计算梯度
    # 若 B > 1, 我们对所有样本分别取均值再 backward, 或逐个样本循环
    grad_outputs = torch.ones_like(scores)
    model.zero_grad()
    scores.backward(gradient=grad_outputs)

    # x.grad 即为相对于输入图像 x 的梯度
    gradient = x.grad.data

    # 通道方向取绝对值再做聚合(如求范数或通道求和)
    # 此处演示对通道方向做绝对值并取最大, 以得到简单的显著图
    saliency_map, _ = gradient.abs().max(dim=1, keepdim=True)

    # 也可裁剪/归一化到 [0,1] 方便可视化
    # 这里仅示例简单归一化
    min_val, max_val = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - min_val) / (max_val - min_val + 1e-10)

    # 返回形状为 [B, 1, H, W], 若只保留空间维度可调用 .squeeze(1)
    return saliency_map