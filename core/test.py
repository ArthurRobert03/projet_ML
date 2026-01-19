from my_models.load_model import get_model
import utils.utils as utils
import torch
import matplotlib.pyplot as plt
import numpy as np

device = "cuda"

model = get_model()


x = torch.randn(1, 512, device=device, dtype=torch.float32, requires_grad=True)

gen = model.test(x)





