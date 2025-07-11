import os
import cv2
import tiffile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Training.mask import random_patch_mask
from Utilities.utils import save_model

def train_model(model, learning_rate=1e-3, input, path, num_iter=1, patch_size=1, mask_ratio=0.2):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  input = input.to(device)
  model.train()
  
  loss_history = []

  os.makedirs(path, exist_ok=True)

  num_iter = 100
  
  for it in range(num_iter):
     masked_input, mask = random_patch_mask(input, patch_size=patch_size, mask_ratio=mask_ratio)
     output = model(masked_input)

     loss = F.mse_loss(output * (1 - mask), noisy_image_tensor * (1-mask))
     loss_history.append(loss.item())

     optimizer.zero_grad()
     loss.backward()
     optimizer.step()

     save_model(it, loss, model, optimizer, path, input)
    

  np.save(path+'loss_history.npy', np.array(loss_history))
