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

# Assurez-vous d'avoir model.py et resnet.py dans le même dossier
from my_models.model import BiSeNet 
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

class FaceParsingManager:
    def __init__(self, weight_path='79999_iter.pth', device='cuda'):
        self.device = device
        self.net = BiSeNet(n_classes=19)
        self.net.to(device)
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Impossible de trouver les poids : {weight_path}")
            
        self.net.load_state_dict(torch.load(weight_path, map_location=device))
        self.net.eval()
        
        # Moyenne/Std utilisés par ce modèle spécifique (ImageNet style)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # Indices des parties du visage pour zllrunning/face-parsing
        # 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eye_g, 
        # 10: nose, 11: mouth, 12: u_lip, 13: l_lip
        # On exclut souvent : 0 (background), 17 (hair), 16 (neck), 18 (cloth) pour l'inversion pure du visage
        self.face_parts = [1, 2, 3, 4, 5, 6,7,8, 9, 10, 11, 12, 13, 14, 15, 17, 16]

    def get_mask(self, img_tensor, include_hair=False):
        """
        img_tensor: (1, 3, H, W) range [-1, 1] ou [0, 1]
        returns: (1, 3, H, W) binary mask (0.0 or 1.0)
        """
        # 1. Prétraitement : Redimensionner à 512x512 et normaliser
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        
        # Passage en [0, 1] si nécessaire
        x = img_tensor.clone()
        if x.min() < 0:
            x = (x + 1) / 2
        
        # Interpolation vers 512x512 (taille requise par BiSeNet)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        
        # Normalisation spécifique (x - mean) / std
        x = (x - self.mean) / self.std

        # 2. Inférence
        with torch.no_grad():
            out = self.net(x)[0] # (1, 19, 512, 512)
            parsing = out.argmax(1) # (1, 512, 512) - Indices des classes

        # 3. Construction du masque
        # On crée un tensor de zéros
        mask = torch.zeros_like(parsing).float()
        
        # On active les parties du visage
        parts = self.face_parts + ([17] if include_hair else []) # 17 = Hair
        
        for part_idx in parts:
            mask = mask + (parsing == part_idx).float()
            
        # Clip au cas où (parfois lèvres et bouche se chevauchent dans la map)
        mask = mask.clamp(0, 1)

        # 4. Redimensionnement à la taille d'origine et format (1, 3, H, W)
        mask = mask.unsqueeze(1) # (1, 1, 512, 512)
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask = mask.repeat(1, 3, 1, 1) # (1, 3, H, W)

        return mask




