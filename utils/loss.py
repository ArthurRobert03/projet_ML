import torch.nn.functional as F

def multiscale_loss(gen, tgt, alphas, mask=None, mode="area"):
    loss = 0.0
    for s, a in alphas.items():
        # Redimensionner l'image et la cible
        g = F.interpolate(gen, size=(s, s), mode=mode)
        t = F.interpolate(tgt, size=(s, s), mode=mode)  
        diff = (g - t).abs() + 10.0 * ((g - t) ** 2)
        if mask is not None:
            m = F.interpolate(mask, size=(s, s), mode="nearest")
            diff = diff * m
            
        loss_s = diff.mean()
        loss = loss + a * loss_s
    return loss