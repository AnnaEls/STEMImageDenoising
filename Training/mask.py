import torch

def random_patch_mask(x, patch_size=1, mask_ratio=0.2):
    """
    Create a soft random patch mask for self-supervised training.

    Args:
        x (torch.Tensor): Input image tensor of shape [B, C, H, W].
        patch_size (int): Size of the random patches to mask.
        mask_ratio (float): Ratio of the total area to mask (between 0 and 1).

    Returns:
        masked_x (torch.Tensor): The masked input image.
        mask (torch.Tensor): mask applied.
    """

    B, C, H, W = x.shape

    # Start with a mask full of ones (no masking)
    mask = torch.ones((B, 1, H, W), device=x.device)

    # Number of patches to mask
    num_patches = int(H * W * mask_ratio / (patch_size * patch_size))

    for _ in range(num_patches):
        top = torch.randint(0, H - patch_size, (1,)).item()
        left = torch.randint(0, W - patch_size, (1,)).item()
        mask[:, :, top:top+patch_size, left:left+patch_size] = 0

    # Clamp values to [0, 1] just in case
    mask = mask.clamp(0, 1)

    # Apply the mask softly
    masked_x = x * mask

    return masked_x, mask
