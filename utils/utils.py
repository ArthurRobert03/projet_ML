import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

def load_img_tensor(path, device):
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).to(device)
    return img

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    return img


def plot_img(img):
    plt.imshow(img)
    plt.show()


def load_img_tensor(path, device, size = 512):
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"PIL failed to open image:\n{path}") from e

    img = img.resize((size, size), resample=Image.BICUBIC)

    img = torch.from_numpy(np.array(img)).float().to(device) / 255.0
    img = img.permute(2, 0, 1)  # (3,H,W)
    return img


def normalize_01(x, eps=1e-8):
    minv = x.amin(dim=(1,2,3), keepdim=True)
    maxv = x.amax(dim=(1,2,3), keepdim=True)
    return (x - minv) / (maxv - minv + eps)





