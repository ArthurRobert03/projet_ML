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

def multiscale_loss(gen, tgt, alphas, mode="area"):
    """
    gen,tgt: (N,3,512,512) dans le même espace (ici gen_vis et img)
    alphas: dict {size: poids}
    """
    loss = 0.0
    for s, a in alphas.items():
        g = F.interpolate(gen, size=(s, s), mode=mode)
        t = F.interpolate(tgt, size=(s, s), mode=mode)
        # tu peux choisir L1 / L2, ici je garde ton mix
        loss_s = (g - t).abs().mean() + 10.0 * ((g - t) ** 2).mean()
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
    # phases (à toi de régler)
    if step < 1000:
        return {16:1.0, 64:0.8, 128:0.3, 256:0.1, 512:0.0}
    elif step < 3000:
        return {16:0.5, 64:0.8, 128:0.6, 256:0.3, 512:0.1}
    elif step < 6000:
        return {16:0.2, 64:0.5, 128:0.7, 256:0.7, 512:0.5}
    else:
        return {16:0.1, 64:0.2, 128:0.4, 256:0.7, 512:1.0}
# ---------- model ----------
model = get_model()
model = model.netG
model.eval()

img = utils.load_img_tensor("data/test.jpg", device).unsqueeze(0)  # (1,3,512,512)

x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)

best_loss = float('inf')
x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
for k in range(1000):
    z = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)
    gen = model(z)                 # (1,3,512,512) brut
    gen_vis = utils.normalize_01(gen)  # (1,3,512,512) en [0,1] (comme toi)

    # multi-scale loss (avec alpha par résolution)
    alphas = get_alphas(10000)
    loss = multiscale_loss(gen_vis, img, alphas, mode="area")
    if loss < best_loss:
        best_loss = loss
        x = z

optimizer = torch.optim.Adam([x], lr=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# résolutions + poids (alpha par résolution)
alphas = {
    512: 1.0,
    256: 0.7,
    128: 0.6,
    64:  0.6,
    16:  0.6
}

for step in range(10000):
    optimizer.zero_grad()

    gen = model(x)                 # (1,3,512,512) brut
    gen_vis = utils.normalize_01(gen)  # (1,3,512,512) en [0,1] (comme toi)

    # multi-scale loss (avec alpha par résolution)
    alphas = get_alphas(step)
    loss = multiscale_loss(gen_vis, img, alphas, mode="area")
    loss.backward()

    optimizer.step()
    scheduler.step()

    if step % 1 == 0:
        print(f"step {step} | loss = {loss.item():.4f}")

        # --- affichage : chaque résolution empilée verticalement (target / gen / diff) ---
        with torch.no_grad():
            tgts = {s: F.interpolate(img, size=(s,s), mode="area") for s in alphas.keys()}
            gens = {s: F.interpolate(gen_vis, size=(s,s), mode="area") for s in alphas.keys()}

        # tensors -> numpy RGB [0,1]
        tgts_np = {s: tgts[s][0].detach().cpu().permute(1,2,0).numpy() for s in tgts}
        gens_np = {s: gens[s][0].detach().cpu().permute(1,2,0).numpy() for s in gens}
        diff_np = {s: np.abs(tgts_np[s] - gens_np[s]) for s in tgts_np}

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
