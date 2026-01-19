from my_models.load_model import get_model
import utils.utils as utils
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = "cuda"

import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

#vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16].eval().to(device)

vgg = vgg16(weights=VGG16_Weights).features[:16].eval().to(device)



def vgg_preprocess(x):  # x in [-1,1], NCHW
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std

def perceptual_loss(gen, tgt):
    gen_f = vgg(gen)
    tgt_f = vgg(tgt)
    return F.l1_loss(gen_f, tgt_f)



model = get_model()
model = model.netG
model.eval()


img = utils.load_img_tensor("data/Camar.jpg", device)
img = img.unsqueeze(0)

x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.Adam([x], lr=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)


gen = model(x)
res = gen[0].cpu().permute(1,2,0).detach().numpy()


for step in range(10000):
    optimizer.zero_grad()

    # forward (doit garder le graphe)
    gen = model(x) # 1x3x512x512, supposé [-1,1]
    gen_vis = utils.normalize_01(gen)
    # loss pixel (L1 est souvent mieux que MSE en visu)
    #loss =  perceptual_loss(gen_vis, img)#+((gen_vis-img)**2).mean()*100 
    loss = ((gen_vis-img)**2).mean()
    loss.backward()

    optimizer.step()
    scheduler.step()

    if step % 1 == 0:
        print(f"step {step} | loss = {loss.item():.4f}")



        image = gen_vis[0].cpu().permute(1,2,0).detach().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        target = img[0].cpu().permute(1,2,0).detach().numpy()
        target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        diff = cv2.absdiff(target, image)
        
        image_final = cv2.hconcat([target,image, diff])
        cv2.putText(image_final, str(float(loss.item())),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("test", image_final)
        if (cv2.waitKey(1)==27):
            break
            




with torch.no_grad():
    res = model(x)
    res = utils.normalize_01(res)

print(res.shape)
res = res[0].cpu().permute(1,2,0).numpy()
utils.plot_img(res)



