import numpy as np
import torch
from PIL import Image

def vertical_flip(x):
        return x.flip(dims=(2,))

def horizontal_flip(x):
    return x.flip(dims=(3,))

if __name__ == '__main__':
    # Test.py
    #加载一张图片，对图片进行vertical_flip操作，对比操作前后的图片
    img = Image.open('data\data\images\ILSVRC2012_val_00000012.JPEG')
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).float()
    img = img.unsqueeze(0)
    img = vertical_flip(img)
    img = img.squeeze(0)
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(img.astype(np.uint8))
    img.show()
    img.save('vertical_flip.jpg')
    #加载一张图片，对图片进行horizontal_flip操作，对比操作前后的图片
    img = Image.open('data\data\images\ILSVRC2012_val_00000012.JPEG')
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).float()
    img = img.unsqueeze(0)
    img = horizontal_flip(img)
    img = img.squeeze(0)
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(img.astype(np.uint8))
    img.show()
    

