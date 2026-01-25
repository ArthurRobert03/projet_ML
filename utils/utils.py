import cv2
import matplotlib.pyplot as plt
import torch

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    return img


def plot_img(img):
    plt.imshow(img)
    plt.show()


def load_img_tensor(path, device):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    img = torch.tensor(img).to(device)
    img = img/255.0
    img = img.permute(2,0,1)
    return img


def normalize_01(x, eps=1e-8):
    minv = x.amin(dim=(1,2,3), keepdim=True)
    maxv = x.amax(dim=(1,2,3), keepdim=True)
    return (x - minv) / (maxv - minv + eps)






