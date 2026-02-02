import utils.utils as utils
import my_models.load_model as model_loader
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
import utils.loss as loss_criterion

with open("config/config.json", "r") as f:
    json_data = json.load(f)
device = json_data["device"]

class Inversor:

    def __init__(self, img):
        self.model = model_loader.get_model()
        self.model = self.model.netG
        self.model.eval()
        self.img_target = (img*2)-1
        self.parser = model_loader.FaceParsingManager(weight_path='my_models/79999_iter.pth', device=device)
        self.face_mask = self.parser.get_mask(img)
        self.x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.x], lr=0.05)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        self.alphas = {64:0.2, 512:0.8}

    def random_start(self, n_random = 50):
        
        best_loss = float('inf')
        for k in range(n_random):
            z = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
            gen = self.model(z)                
            gen_vis = utils.normalize_01(gen)  

            alphas = {64:1}
            loss = loss_criterion.multiscale_loss(gen_vis, self.img_target, alphas, mask = self.face_mask, mode="area")
            if loss.item() < best_loss:
                best_loss = loss
                self.x = z
        self.optimizer = torch.optim.Adam([self.x], lr=0.05)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)


    def step(self):
        self.optimizer.zero_grad()
        gen = self.model(self.x)                 
        gen_vis = utils.normalize_01(gen)  
        loss = loss_criterion.multiscale_loss(gen, self.img_target, self.alphas,mask = self.face_mask, mode="area")
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), gen_vis[0].detach().cpu().permute(1,2,0).numpy()

    def toggle_mask(self):
        self.mask = not self.mask

    def get_mask(self):
        return self.face_mask

    def set_alphas(self, alphas):
        self.alphas = alphas
