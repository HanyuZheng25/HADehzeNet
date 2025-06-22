import cv2
import numpy as np
import torch
from torch.nn import functional as F

"""
这段代码实现了使用增强的 USM 锐化方法对图像进行锐化处理，其中包括了 Numpy 和 PyTorch 两个版本的实现。

### Numpy 实现

- `usm_sharp(img, weight=0.5, radius=50, threshold=10)`: 对输入图像进行 USM 锐化处理。
  - `img`: 输入图像，是一个 HWC 形状的 Numpy 数组，表示图像的通道顺序为 BGR，像素值范围在 [0, 1]。
  - `weight`: 锐化强度权重，默认为 0.5。
  - `radius`: 高斯模糊的卷积核大小，默认为 50。
  - `threshold`: 阈值，用于计算掩码，控制锐化区域的范围，默认为 10。
  
### PyTorch 实现

- `class USMSharp(torch.nn.Module)`: 继承自 `torch.nn.Module`，实现了 USM 锐化的 PyTorch 模块。
  - `__init__(self, radius=50, sigma=0)`: 初始化方法，设置高斯模糊的卷积核。
    - `radius`: 高斯模糊的卷积核大小，默认为 50。
    - `sigma`: 高斯模糊的标准差，默认为 0。
  - `forward(self, img, weight=0.5, threshold=10)`: 前向传播方法，实现 USM 锐化处理。
    - `img`: 输入图像，是一个 BCHW 形状的 PyTorch 张量，表示图像的通道顺序为 RGB，像素值范围在 [0, 1]。
    - `weight`: 锐化强度权重，默认为 0.5。
    - `threshold`: 阈值，用于计算掩码，控制锐化区域的范围，默认为 10。

这两个实现的核心思想都是对图像进行高斯模糊处理，然后计算图像与模糊图像之间的残差，根据残差生成掩码，最后将掩码应用到锐化后的图像上，以获得锐化效果。
"""
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img
