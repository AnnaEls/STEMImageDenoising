import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

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
  #noisy_image_tensor = torch.from_numpy(z_score_normalize(noisy_image)).unsqueeze(0).unsqueeze(0).float()
  noisy_image_tensor = torch.from_numpy(noisy_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
  noisy_image_tensor = z_score_normalize(noisy_image_tensor)
  if show_image:
     plt.imshow(noisy_image_tensor[0,0], cmap='gray'); plt.axis('off'); plt.tight_layout(); plt.show();
  return noisy_image_tensor

def convert(image):
  image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
  return image

def save_model(it, loss, model, optimizer, path, noisy_image_tensor, show_image = True, model_eval = True):
  
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
               print(f"epoch {it + 1}, loss={loss.item():.6f}")
               plt.imshow(denoised_image.squeeze().detach().cpu().numpy(), cmap='gray'); plt.axis('off'); plt.tight_layout();
               plt.show()
     model.train()

def calculate_metrics(path, path_to_clean_image, show_graphs=False):
    clean_image = np.array(tifffile.imread(path_to_clean_image))
    psnrs = []
    ssims = []
    iterations = []
    image_files = sorted([f for f in os.listdir(path) if f.endswith('.tif') and f[:-4].isdigit()])
    for image_file in image_files:
        iteration = int(image_file[:-4])
        denoised_image = np.array(tifffile.imread(os.path.join(path, image_file)))
        current_psnr = psnr_metric(clean_image, denoised_image)
        current_ssim = ssim_metric(clean_image, denoised_image)
        psnrs.append(current_psnr)
        ssims.append(current_ssim)
        iterations.append(iteration)
    if show_graphs:
        sorted_indices = np.argsort(iterations)
        iterations = np.array(iterations)[sorted_indices]
        psnrs = np.array(psnrs)[sorted_indices]
        ssims = np.array(ssims)[sorted_indices]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, psnrs)
        plt.xlabel('Iterations')
        plt.ylabel('PSNR')
        plt.title('PSNR vs. Iterations')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(iterations, ssims)
        plt.xlabel('Iterations')
        plt.ylabel('SSIM')
        plt.title('SSIM vs. Iterations')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    max_psnr = np.max(psnrs)
    max_ssim = np.max(ssims)
    best_psnr_iteration = iterations[np.argmax(psnrs)]
    best_ssim_iteration = iterations[np.argmax(ssims)]
    np.save(os.path.join(path, 'psnrs.npy'), psnrs)
    np.save(os.path.join(path, 'ssims.npy'), ssims)
    
    best_psnr_model_file = f"model_step_{best_psnr_iteration+1:04d}.pt"
    best_ssim_model_file = f"model_step_{best_ssim_iteration+1:04d}.pt"
    model_files = [f for f in os.listdir(path) if f.endswith('.pt')]
    for model_file in model_files:
        if model_file != best_psnr_model_file and model_file != best_ssim_model_file:
           try:
              os.remove(os.path.join(path, model_file))
           except OSError as e:
              print(f"Error deleting {model_file}: {e}")
    return max_psnr, max_ssim, best_psnr_iteration, best_ssim_iteration
