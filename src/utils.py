import torch
import cv2
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_image_tensor(tensor, path):
    """
    Saves a (C,H,W) tensor as an image.
    Tensor values should be in [0, 1].
    """
    img_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1) * 255.0
    img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
