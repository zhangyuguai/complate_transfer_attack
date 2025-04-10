import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import torch
import lpips
from torchvision import transforms


def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img


# 计算SSIM
def compute_ssim(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)

    ssim_value = ssim(image1, image2, multichannel=True, win_size=3)
    return ssim_value


# 计算PSNR
def compute_psnr(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)

    psnr_value = psnr(image1, image2)
    return psnr_value


# 计算LPIPS
def compute_lpips(image1, image2, model):
    transform = transforms.Compose([transforms.ToTensor()])
    image1 = transform(image1).unsqueeze(0)
    image2 = transform(image2).unsqueeze(0)

    device = next(model.parameters()).device
    image1 = image1.to(device)
    image2 = image2.to(device)

    with torch.no_grad():
        lpips_value = model(image1, image2)
    return lpips_value.item()



# 计算所有图片对的平均SSIM, PSNR 和 LPIPS
def compute_average_metrics(output_dir, lpips_model):
    ssim_values = []
    psnr_values = []
    lpips_values = []

    for img_name in os.listdir(output_dir):
        if not img_name.endswith('_adv_image.png'):
            continue

        base_name = img_name.replace('_adv_image.png', '_originImage.png')
        adv_img_path = os.path.join(output_dir, img_name)
        original_img_path = os.path.join(output_dir, base_name)

        if not os.path.exists(original_img_path):
            print(f"警告：原始图片 {original_img_path} 不存在，跳过该文件。")
            continue

        adv_img = preprocess_image(adv_img_path)
        original_img = preprocess_image(original_img_path)

        # 计算SSIM
        ssim_value = compute_ssim(adv_img, original_img)
        ssim_values.append(ssim_value)

        # 计算PSNR
        psnr_value = compute_psnr(adv_img, original_img)
        psnr_values.append(psnr_value)

        # 计算LPIPS
        lpips_value = compute_lpips(adv_img, original_img, lpips_model)
        lpips_values.append(lpips_value)

    # 计算平均值
    average_ssim = np.mean(ssim_values) if ssim_values else 0
    average_psnr = np.mean(psnr_values) if psnr_values else 0
    average_lpips = np.mean(lpips_values) if lpips_values else 0

    return average_ssim, average_psnr, average_lpips

output_dir = r'E:\python_workspace\TransferAttack-main\temp'
lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')
average_ssim, average_psnr, average_lpips = compute_average_metrics(output_dir, lpips_model)
print(f"{output_dir} SSIM: {average_ssim:.4f}, PSNR: {average_psnr:.4f}, LPIPS: {average_lpips:.4f}")

