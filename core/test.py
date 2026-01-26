from my_models.load_model import get_model
import utils.utils as utils
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = "cuda"
import torch.nn.functional as F


# ---------- multi-scale helpers ----------

def downscale_pyramid(img, sizes=(256, 128, 64, 32), mode="area"):
    """
    img: (N,3,H,W)
    retourne dict {size: (N,3,size,size)}
    """
    return {s: F.interpolate(img, size=(s, s), mode=mode) for s in sizes}


def create_face_mask(img_tensor, pad_ratio=0.1):
    """
    Crée un masque binaire (1 sur le visage, 0 ailleurs).
    img_tensor: (1, 3, H, W) sur GPU ou CPU, range quelconque
    Retourne: (1, 3, H, W) tensor binaire sur le même device
    """
    # 1. Conversion Tensor -> Numpy uint8 pour OpenCV
    # On suppose que l'image est normalisée, on la remet en [0, 255]
    im_temp = img_tensor.detach().cpu()
    if im_temp.min() < 0: # Si range [-1, 1]
        im_temp = (im_temp + 1) / 2
    
    im_np = im_temp[0].permute(1, 2, 0).numpy()
    im_np = (im_np * 255).astype(np.uint8)
    
    # 2. Détection de visage (Haar Cascade)
    # Ce fichier xml est inclus dans cv2 par défaut
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(im_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 3. Création du masque
    mask = np.zeros_like(gray, dtype=np.float32) # Fond noir

    if len(faces) > 0:
        # On prend le plus grand visage trouvé
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        # On dessine une ellipse (plus naturel qu'un rectangle pour un visage)
        center = (x + w//2, y + h//2)
        axes = (int(w/2 * (1+pad_ratio)), int(h/2 * (1+pad_ratio))) # Un peu plus large
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1) 
    else:
        print("Attention : Aucun visage détecté, le masque est plein (1.0).")
        mask[:] = 1.0 # Fallback : on prend tout si pas de visage

    # 4. Conversion Numpy -> Tensor (N, C, H, W)
    # On duplique le masque sur les 3 canaux de couleur pour faciliter la multiplication
    mask_t = torch.from_numpy(mask).to(img_tensor.device)
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    
    return mask_t


def multiscale_loss(gen, tgt, alphas, mask=None, mode="area"):
    loss = 0.0
    for s, a in alphas.items():
        # Redimensionner l'image et la cible
        g = F.interpolate(gen, size=(s, s), mode=mode)
        t = F.interpolate(tgt, size=(s, s), mode=mode)
        
        diff = (g - t).abs() + 10.0 * ((g - t) ** 2)
        
        # --- APPLIQUER LE MASQUE ---
        if mask is not None:
            # On redimensionne le masque à la taille s (Nearest pour garder 0 ou 1 pur, ou Area pour du soft)
            m = F.interpolate(mask, size=(s, s), mode="nearest")
            
            # On multiplie la différence par le masque
            # Les pixels hors visage (0) ne compteront pas dans la loss
            diff = diff * m
            
        loss_s = diff.mean()
        loss = loss + a * loss_s
    return loss


    

def stack_for_display(imgs_by_res, bgr=True, pad_value=0):
    """
    imgs_by_res: dict {res: numpy HxWx3 en [0,1]}
    retourne image uint8 empilée verticalement, en paddant à la largeur max
    """
    res_list = sorted(imgs_by_res.keys(), reverse=True)  # 512 en haut
    rows = []

    # largeur max (celle de la plus grande résolution)
    max_w = max(imgs_by_res[r].shape[1] for r in res_list)

    for r in res_list:
        im = imgs_by_res[r]
        im = (im * 255.0).clip(0, 255).astype(np.uint8)

        if bgr:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        h, w, c = im.shape
        if w < max_w:
            pad = np.full((h, max_w - w, c), pad_value, dtype=im.dtype)
            im = cv2.hconcat([im, pad])   # pad à droite

        rows.append(im)

    return cv2.vconcat(rows)

def get_alphas(step):
    if step < 500:
        return {16:0.9, 64:0.1, 512:0.0}
    elif step < 1000:
        return {16:0.2, 64:0.7, 512:0.1}
    elif step < 3000:
        return {16:0.0, 64:0.4, 512:0.6}
    else:
        return {16:0.0, 64:0.1, 512:0.9}
# ---------- model ----------
model = get_model()
model = model.netG
model.eval()
#MTCNN
img = utils.load_img_tensor("data/48.png", device).unsqueeze(0)  # (1,3,512,512)
#img_masque = utils.load_img_tensor("data/39_masque.png", device).unsqueeze(0)
img_target = (img*2)-1



parser = utils.FaceParsingManager(weight_path='my_models/79999_iter.pth', device=device)

# Création du masque (include_hair=False pour se focaliser sur le visage seul)
face_mask = parser.get_mask(img, include_hair=False)

# Vérification visuelle du masque au démarrage

x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)

best_loss = float('inf')
x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
for k in range(50):
    z = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
    gen = model(z)                 # (1,3,512,512) brut
    gen_vis = utils.normalize_01(gen)  # (1,3,512,512) en [0,1] (comme toi)

    # multi-scale loss (avec alpha par résolution)
    alphas = get_alphas(0)
    loss = multiscale_loss(gen_vis, img, alphas, mask = face_mask, mode="area")
    if loss < best_loss:
        best_loss = loss
        x = z

optimizer = torch.optim.Adam([x], lr=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

alphas = {
    512: 1.0,
    64:  0.6,
    16:  0.6
}

for step in range(10000):
    optimizer.zero_grad()

    gen = model(x)                 # (1,3,512,512) brut
    gen_vis = utils.normalize_01(gen)  # (1,3,512,512) en [0,1] (comme toi)

    # multi-scale loss (avec alpha par résolution)
    alphas = get_alphas(step)
    loss = multiscale_loss(gen, img_target, alphas,mask = face_mask, mode="area")
    loss.backward()

    optimizer.step()
    scheduler.step()

    if step % 1 == 0:
        print(f"step {step} | loss = {loss.item():.4f}")

        # --- affichage : chaque résolution empilée verticalement (target / gen / diff) ---
        with torch.no_grad():
            tgts = {s: F.interpolate(img*0.3+img*face_mask, size=(s,s), mode="area") for s in alphas.keys()}
            gens = {s: F.interpolate(gen_vis, size=(s,s), mode="area") for s in alphas.keys()}
            diffs = {s: F.interpolate(gen_vis*face_mask, size=(s,s), mode="area") for s in alphas.keys()}

        # tensors -> numpy RGB [0,1]
        tgts_np = {s: tgts[s][0].detach().cpu().permute(1,2,0).numpy() for s in tgts}
        gens_np = {s: gens[s][0].detach().cpu().permute(1,2,0).numpy() for s in gens}
        diff_np = {s: diffs[s][0].detach().cpu().permute(1,2,0).numpy() for s in diffs}

        panel_t = stack_for_display(tgts_np, bgr=True)
        panel_g = stack_for_display(gens_np, bgr=True)
        panel_d = stack_for_display(diff_np, bgr=True)

        panel = cv2.hconcat([panel_t, panel_g, panel_d])
        cv2.putText(panel, f"loss={float(loss.item()):.4f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("multiscale", panel)
        if cv2.waitKey(1) == 27:
            break


with torch.no_grad():
    res = utils.normalize_01(model(x))

print(res.shape)
res_np = res[0].cpu().permute(1,2,0).numpy()
utils.plot_img(res_np)
