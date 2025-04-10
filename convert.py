import os
import re
import shutil

def rename_original_images(source_dir, target_dir):
    """
    将ILSVRC2012格式的原始图像文件重命名为指定的新格式
    
    参数:
        source_dir: 包含原始图像文件的目录
        target_dir: 保存重命名后文件的目录
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 正则表达式匹配ILSVRC2012_val_00000001.JPEG格式的文件名
    pattern = re.compile(r'ILSVRC2012_val_(\d+)\.JPEG')
    
    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        if match:
            # 提取数字部分并去除前导零
            number_str = match.group(1)
            number_without_zeros = str(int(number_str))
            
            # 创建新的文件名
            new_name = f"{number_without_zeros}_originImage.png"
            
            # 复制并重命名原始文件
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_name)
            shutil.copy2(source_path, target_path)
            
            print(f"已重命名原始图像: {filename} -> {new_name}")


def rename_adversarial_images(source_dir, target_dir, adv_suffix="_adv"):
    """
    将ILSVRC2012格式的对抗样本图像文件重命名为指定的新格式
    
    参数:
        source_dir: 包含对抗样本图像文件的目录
        target_dir: 保存重命名后文件的目录
        adv_suffix: 对抗样本文件名中的后缀，默认为"_adv"
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    print("已创建目标目录")
    # 正则表达式匹配对抗样本文件名，假设格式为ILSVRC2012_val_00000001_adv.JPEG
    pattern = re.compile(fr'ILSVRC2012_val_(\d+){adv_suffix}\.JPEG')
    
    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        if match:
            # 提取数字部分并去除前导零
            number_str = match.group(1)
            number_without_zeros = str(int(number_str))
            
            # 创建新的文件名
            new_name = f"{number_without_zeros}_adv_originImage.png"
            
            # 复制并重命名对抗样本文件
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_name)
            shutil.copy2(source_path, target_path)
            
            print(f"已重命名对抗样本: {filename} -> {new_name}")

# 使用示例
# rename_original_images(r'E:\python_workspace\TransferAttack-main\data\val_rs', r'E:\python_workspace\TransferAttack-main\temp')
rename_original_images(r'E:\python_workspace\TransferAttack-main\data\val_rs', r'E:\python_workspace\TransferAttack-main\temp')