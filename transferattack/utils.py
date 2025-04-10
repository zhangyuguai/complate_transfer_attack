import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import timm
import os

from transferattack.nets.inceptionresnetv2 import InceptionResNetV2
from transferattack.nets.inceptionv4 import InceptionV4

img_height, img_width = 224, 224
img_max, img_min = 1., 0

cnn_model_paper = ['mobilenet_v2', 'inception_v3','densenet121','resnet18','resnet50','resnet152','resnet101','vgg16','vgg19']
timm_cnn_model = ['inception_v4','inception_resnet_v2']
tf_ens_adv_in1k =['inception_resnet_v2']
tf_in1k = ['inception_v4']
# vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
#                    'visformer_small', 'swin_tiny_patch4_window7_224']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
                   'visformer_small', 'swin_tiny_patch4_window7_224']
cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

def load_pretrained_model(cnn_model=[], vit_model=[]):
    # print(timm.list_models('*inception_resnet*'))
    for model_name in timm_cnn_model:
        yield model_name, timm.create_model(model_name, pretrained=True)
    for model_name in cnn_model:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")
    #     yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")


    

    #加载两个特别模型
    # inception_v4 = InceptionV4(num_classes=1000)
    # inception_v4.load_state_dict(torch.load('./checkpoints/inception_v4.bin',weights_only=True))
    # yield 'inception_v4', inception_v4
    # inception_resnet_v2 =InceptionResNetV2(num_classes=1000)
    # inception_resnet_v2.load_state_dict(torch.load('./checkpoints/inception_resnet_v2.bin',weights_only=True))
    # yield 'inception_resnet_v2', inception_resnet_v2

    # model = timm.create_model('inception_resnet_v2.tf_in1k', pretrained=True)
    # model = model.eval()
    # yield 'inception_resnet_v2', model
    for model_name in vit_model:
        yield model_name, timm.create_model(model_name, pretrained=True)

def adjust_labels(labels):
    """
    将标签从 1-based 转换为 0-based（如果需要）。
    """
    if labels.min() == 1:  # 如果标签从 1 开始
        logging.info("Detected 1-based labels. Converting to 0-based.")
        labels = labels - 1
    return labels

def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    model_name = model.__class__.__name__
    Resize = 224
    
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        if 'Inc' in model_name:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            Resize = 299
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            Resize = 224

    PreprocessModel = PreprocessingModel(Resize, mean, std)
    return torch.nn.Sequential(PreprocessModel, model)


def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize)
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'val_rs.csv'))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]

        assert isinstance(filename, str)

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            if self.target_class:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], self.target_class] for i in range(len(dev))}
            else:
                f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l




if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=False, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
