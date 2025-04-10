import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import scipy.stats as st

from ..utils import *
from ..gradient.mifgsm import MIFGSM

class IDAA(MIFGSM):
    """
    IDAA(Input-Diversity-based Adaptive Attack)
    'Boosting the Transferability of Adversarial Examples via Local Mixup and Adaptive Step Size'(https://arxiv.org/pdf/2401.13205)

    改动要点:
    1) 使用 Adam 样式的一阶 (mg) 和二阶 (vg) 动量替代传统动量, 以实现自适应步长。
    2) 使用线性插值 (local mixup) 替代简单的覆盖粘贴, 使局部块随机融合。
    """

    def __init__(
            self,
            model_name,
            epsilon=0.07,
            alpha=1,
            epoch=10,
            decay=1.0,
            num_scale=20,
            num_block=3,
            crop_size=0.7,
            targeted=True,
            random_start=False,
            norm='linfty',
            loss='crossentropy',
            device=None,
            attack='IDAA',
            **kwargs
    ):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block
        self.kernel = self.gkern()
        self.crop_size = crop_size

        # 原论文里借鉴了 Adam 优化, 这里 beta1, beta2, eps 用于自适应步长
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps_adam = 1e-8

        # 替换所有数据增强操作
        self.op = [
            self.vertical_shift,
            self.horizontal_shift,
            self.vertical_flip,
            self.horizontal_flip,
            self.rotate180,
            self.scale,
            self.add_noise
        ]

    # ----------------- 数据增强操作示例 -------------------
    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x):
        return torch.rand(1)[0] * x

    def add_noise(self, x):
        return torch.clamp(x + torch.zeros_like(x).uniform_(-16/255, 16/255), 0, 1)

    # ----------------- 高斯模糊内核 -------------------
    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def blur(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding='same', groups=3)

    # ----------------- 随机分块变换, 并对每块应用随机操作 -------------------
    def blocktransform(self, x):
        _, _, w, h = x.shape
        y_axis = [0] + np.random.choice(range(1, h), self.num_block - 1, replace=False).tolist() + [h]
        x_axis = [0] + np.random.choice(range(1, w), self.num_block - 1, replace=False).tolist() + [w]
        y_axis.sort()
        x_axis.sort()

        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

    # ----------------- transform() 将原图扩增为多份 -------------------
    def transform(self, x, **kwargs):
        return torch.cat([self.blocktransform(x) for _ in range(self.num_scale)], dim=0)

    # ----------------- 计算损失 (针对多份图像) -------------------
    def get_loss(self, logits, label):
        # label.repeat(self.num_scale) 对应 transform 扩增后的 batch
        return (
            -self.loss(logits, label.repeat(self.num_scale))
            if self.targeted
            else self.loss(logits, label.repeat(self.num_scale))
        )

    # ----------------- 输入与扰动的界限 -------------------
    def get_bound(self, x):
        lower_bound = -torch.min(x, self.epsilon * torch.ones_like(x))
        upper_bound = torch.min(1 - x, self.epsilon * torch.ones_like(x))
        return lower_bound, upper_bound

    # ----------------- 将参数 w 的取值映射到合法范围 (使用 tanh 约束) -------------------
    def compute_perturbation(self, w, lb, ub):
        return lb + (ub - lb) * (torch.tanh(w) / 2 + 1/2)

    # ----------------- IDAA 中自适应步长(Adam风格)替换简单的动量法 -------------------
    def update_delta_adam(self, delta, data, grad, mg, vg, t, base_alpha):
        """
        使用 Adam 样式更新:
            mg = beta1 * mg + (1-beta1) * grad
            vg = beta2 * vg + (1-beta2) * (grad^2)
            m_hat = mg / (1 - beta1^t)
            v_hat = vg / (1 - beta2^t)
            delta <- delta + alpha * m_hat / (sqrt(v_hat) + eps_adam)
        """
        mg = self.beta1 * mg + (1 - self.beta1) * grad
        vg = self.beta2 * vg + (1 - self.beta2) * (grad * grad)

        m_hat = mg / (1 - self.beta1 ** t)
        v_hat = vg / (1 - self.beta2 ** t)

        # 这里 base_alpha 对应最初设置的 self.alpha
        adaptive_step = base_alpha * m_hat / (v_hat.sqrt() + self.eps_adam)
        delta = delta + adaptive_step
        return delta, mg, vg

    # ----------------- 替换原先的 update_delta, 若还需 sign() 可在此调整 -------------------
    def update_delta(self, delta, data, grad, alpha, **kwargs):
        # 保留接口, 但实际在 forward 里调用 update_delta_adam
        return delta + alpha * grad.sign()

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure based on IDAA:
        1) 随机初始化 (如果 random_start=True)
        2) 多分辨率/多变换 (blocktransform)
        3) 在对抗迭代中, 使用两份数据 B1,B2, 随机裁剪并做 local mixup
        4) 用自适应步长(Adam风格)更新扰动
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Bound
        ub, lb = self.get_bound(data)
        # CrossEntropy 的标签需 0-based
        label = label - 1

        # 初始化扰动
        delta = self.init_delta(data)
        # 记录Adam风格的一阶、二阶动量
        mg = torch.zeros_like(delta, device=self.device)
        vg = torch.zeros_like(delta, device=self.device)

        # 确保初始扰动映射到合法范围
        r = self.compute_perturbation(delta, lb, ub)

        crop_H = int(data.shape[2] * self.crop_size)
        crop_W = int(data.shape[3] * self.crop_size)

        # 迭代过程
        pbar = tqdm(range(self.epoch),
                    desc=f'IDAA Attack ({ "Targeted" if self.targeted else "Untargeted" })',
                    leave=True)

        for i in range(1, self.epoch + 1):
            # B1, B2 进行局部混合
            B1 = self.transform(data + self.compute_perturbation(delta, lb, ub))
            B2 = self.transform(data + self.compute_perturbation(delta, lb, ub))

            # 随机裁剪 B2
            start_h = np.random.randint(0, data.shape[2] - crop_H)
            start_w = np.random.randint(0, data.shape[3] - crop_W)
            croped_B2 = B2[:, :, start_h:start_h + crop_H, start_w:start_w + crop_W]

            # 同样随机裁剪 B1
            start_h2 = np.random.randint(0, data.shape[2] - crop_H)
            start_w2 = np.random.randint(0, data.shape[3] - crop_W)

            # 这里使用简单随机 lam, 做 local mixup
            lam = np.random.uniform(0.3, 0.7)
            region_B1 = B1[:, :, start_h2:start_h2 + crop_H, start_w2:start_w2 + crop_W]
            # 线性插值
            B1[:, :, start_h2:start_h2 + crop_H, start_w2:start_w2 + crop_W] = \
                lam * region_B1 + (1 - lam) * croped_B2

            # 前向计算得到 logits
            logits = self.get_logits(B1)
            # 计算 loss
            loss = self.get_loss(logits, label)

            # 计算梯度
            grad = self.get_grad(loss, delta)

            # 使用 Adam 风格自适应步长
            delta, mg, vg = self.update_delta_adam(
                delta=delta,
                data=data,
                grad=grad,
                mg=mg,
                vg=vg,
                t=i,             # 当前迭代步
                base_alpha=self.alpha  # 基准步长
            )

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 最后计算并返回扰动
        return self.compute_perturbation(delta, lb, ub)


