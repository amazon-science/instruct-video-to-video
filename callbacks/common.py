import os
import torch
import torchvision
import wandb
import cv2

def unnorm(x):
    '''convert from range [-1, 1] to [0, 1]'''
    return (x+1) / 2

def clip_image(x, min=0., max=1.):
    return torch.clamp(x, min=min, max=max)

def format_dtype_and_shape(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        if len(x.shape) == 4 and x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        x = x.detach().cpu().numpy()
    return x

def tensor2image(x):
    x = x.float() # handle bf16
    '''convert 4D (b, dim, h, w) pytorch tensor to wandb Image class'''
    grid_img = torchvision.utils.make_grid(
        x, nrow=4
    ).permute(1, 2, 0).detach().cpu().numpy()
    img = wandb.Image(
        grid_img
    )
    return img

def save_figure(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if image.min() < 0:
        image = clip_image(unnorm(image))
    image = format_dtype_and_shape(image)
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(save_path, image[..., ::-1])

def save_sampling_history(image, save_path):
    if image.min() < 0:
        image = clip_image(unnorm(image))
    grid_img = torchvision.utils.make_grid(image, nrow=4)
    save_figure(grid_img, save_path)