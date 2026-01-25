from my_models.load_model import get_model
import utils.utils as utils
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = "cuda"

import torch.nn.functional as F
torch.manual_seed(1)

model = get_model()
model = model.netG
model.eval()


img = utils.load_img_tensor("data/Camar.jpg", device)
img = img.unsqueeze(0)

x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
print(x)
gen = model(x)
gen_vis = utils.normalize_01(gen)
res = gen_vis[0].cpu().permute(1,2,0).detach().numpy()


print(res.shape)
utils.plot_img(res)



