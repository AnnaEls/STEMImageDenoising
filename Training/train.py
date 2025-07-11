import cv2
import tiffile
import matplotlib.pyplot as plt

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
