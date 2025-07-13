import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import os
import cv2

def z_score_normalize(img):
    """
    Z-score normalizes a numpy image (any shape).
    Output will have mean 0 and std 1.
    """
    mean = np.mean(img)
    std = np.std(img)
    # Prevent division by zero
    if std == 0:
        std = 1
    return (img - mean) / std

def prepare_input(path, show_image = True):
  noisy_image = np.array(tifffile.imread(path))
  noisy_image_tensor = torch.from_numpy(z_score_normalize(noisy_image)).unsqueeze(0).unsqueeze(0).float()
  if show_image:
     plt.imshow(noisy_image_tensor[0,0], cmap='gray'); plt.axis('off'); plt.tight_layout(); plt.show();
  return noisy_image_tensor

def convert(image):
  image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
  return image

def save_model(it, loss, model, optimizer, path, noisy_image_tensor, show_image = True, model_eval = True):
  print(f"epoch {it + 1}, loss={loss.item():.6f}")
  torch.save({'step': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, os.path.join(path, f"model_step_{(it+1):04d}.pt"))
  
  if model_eval:
     model.eval()
     with torch.no_grad():
            denoised_image = model(noisy_image_tensor)
            tifffile.imwrite(f'{path}/{it+1:04d}.tif', convert(denoised_image.squeeze().detach().cpu().numpy()), imagej=True)
            if show_image:
               plt.imshow(denoised_image.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               plt.show()
     model.train()
