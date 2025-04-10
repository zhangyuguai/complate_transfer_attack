import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.feature_maps = None
        self.gradients = None
        
        # 注册前向钩子和反向钩子
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(self._forward_hook_function)
                module.register_backward_hook(self._backward_hook_function)
    
    def _forward_hook_function(self, module, input, output):
        self.feature_maps = output
    
    def _backward_hook_function(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
    
    def generate_cam(self, class_idx=None):
        # 若未指定class_idx则取预测最高分类
        if class_idx is None:
            pred = self.model_output.argmax(dim=1).item()
        else:
            pred = class_idx

        one_hot = torch.zeros_like(self.model_output)
        one_hot[0][pred] = 1
        self.model.zero_grad()
        self.model_output.backward(gradient=one_hot, retain_graph=True)
        
        gradients_mean = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = F.relu(torch.sum(self.feature_maps * gradients_mean, dim=1, keepdim=True))
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam[0, 0].detach().cpu().numpy()
    
    def __call__(self, x, class_idx=None):
        self.model_output = self.model(x)
        return self.generate_cam(class_idx=class_idx)

def generate_batch_saliency_maps(
    input_dir="batch_images",
    output_dir="batch_results",
    model_name="resnet18",
    target_layer="layer4.1.conv2",
    image_size=224
):
    """
    从input_dir目录批量读取图像，生成显著性图后保存到output_dir。
    默认使用ResNet18模型，最后一层卷积为层名layer4.1.conv2。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"正在生成显著性图，输入目录：{input_dir}，输出目录：{output_dir}")
    # 加载预训练模型
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError("仅示例ResNet18，如需其他模型请自行扩展。")
    
    model.eval()
    grad_cam = GradCAM(model, target_layer)

    # 定义图像预处理
    transform_fn = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 遍历输入目录的所有图像
    for idx, filename in enumerate(sorted(os.listdir(input_dir))):
        file_path = os.path.join(input_dir, filename)
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")or filename.lower().endswith(".jpeg")):
            continue
        print('正在处理：', file_path)
        # 读取图像
        img = Image.open(file_path).convert("RGB")
        original_width, original_height = img.size

        # 预处理
        input_tensor = transform_fn(img).unsqueeze(0)
        
        # 计算Grad-CAM
        cam_map = grad_cam(input_tensor)

        # 上采样回原图大小
        cam_map_resized = cv2.resize(cam_map, (original_width, original_height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        
        # 原图��热力图叠加
        original_img = np.array(img) / 255.0
        overlay = 0.5 * heatmap + 0.5 * original_img
        
        # 保存结果
        overlay_bgr = np.uint8(255 * overlay)
        out_filepath = os.path.join(output_dir, f"grad_cam_overlay_{idx}.jpg")
        cv2.imwrite(out_filepath, overlay_bgr)
        print(f"已生成显著性图：{out_filepath}")

if __name__ == "__main__":
    # 示例调用：从当前目录下的batch_images文件夹读取图像，并将结果保存到batch_results文件夹中
    input_dir = "E:\\python_workspace\\TransferAttack-main\\data\\images"
    output_dir = "E:\\python_workspace\\TransferAttack-main\\adv_data\\sia\\origin"
    generate_batch_saliency_maps(input_dir=input_dir, output_dir=output_dir)