import torch
from my_models.model import BiSeNet 
import torch
import torch.nn.functional as F
import numpy as np
import os
import json

use_gpu = True if torch.cuda.is_available() else False

with open("config/config.json", "r") as f:
    json_data = json.load(f)
use_gpu = json_data["device"] == "cuda"

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)



def get_model():
    return model


def get_model_256():
    model2 = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-256',
                       pretrained=True, useGPU=use_gpu)


    return model2



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
        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.face_parts = [1, 2, 3, 4, 5, 6,7,8, 9, 10, 11, 12, 13, 14, 15, 17, 16]
        self.part_weight = [0,0.5,0.8,0.8,2,2,3,3,0.8,0.8,1,1,1,1,0.5,0.5,0.2,0.3,0.05]

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
            mask = mask + (parsing == part_idx).float()*self.part_weight[part_idx]
            
        # Clip au cas où (parfois lèvres et bouche se chevauchent dans la map)
        mask = mask.clamp(0, 1)

        # 4. Redimensionnement à la taille d'origine et format (1, 3, H, W)
        mask = mask.unsqueeze(1) # (1, 1, 512, 512)
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        mask = mask.repeat(1, 3, 1, 1) # (1, 3, H, W)

        return mask
    
    def get_mask_grad(self, img_tensor, include_hair=False, temperature=0.7):
        """
        img_tensor: (1,3,H,W) en [-1,1] ou [0,1]
        return: (1,3,H,W) masque soft différentiable wrt img_tensor
        """
        h, w = img_tensor.shape[2], img_tensor.shape[3]

        x = img_tensor
        if x.min() < 0:
            x = (x + 1) / 2  # -> [0,1]

        x = F.interpolate(x, size=(512,512), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std

        # PAS de no_grad, PAS de argmax
        logits = self.net(x)[0]  # (1,19,512,512)
        probs = torch.softmax(logits / temperature, dim=1)  # diff

        parts = self.face_parts + ([17] if include_hair else [])
        mask = 0.0
        for part_idx in parts:
            wgt = float(self.part_weight[part_idx])
            mask = mask + wgt * probs[:, part_idx:part_idx+1]  # (1,1,512,512)

        mask = mask.clamp(0,1)
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        mask = mask.repeat(1,3,1,1)
        return mask



