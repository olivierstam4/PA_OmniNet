import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import math
import torch
def norm_img(x):
    x = x.squeeze().permute(1, 2, 0).detach().numpy()
    return (x - x.min()) / (x.max() - x.min())


def normalize_min_max(image_tensor):
    x_min = image_tensor.min()
    x_max = image_tensor.max()
    normalized_image = (image_tensor - x_min) / (x_max - x_min)
    return normalized_image


def compute_rmse(image, reference):
    mse = F.mse_loss(normalize_min_max(image),
                     normalize_min_max(reference)).item()
    return math.sqrt(mse)


def compute_psnr(image, reference, data_range=1.0):
    mse = F.mse_loss(normalize_min_max(image), normalize_min_max(reference))
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(data_range / torch.sqrt(mse))
    return psnr


# def compute_ssim(image, reference):
#     norm_img = normalize_min_max(image).detach().cpu().numpy()
#     norm_ref = normalize_min_max(reference).detach().cpu().numpy()
#
#     norm_img = norm_img[0].transpose(1, 2, 0)
#     norm_ref = norm_ref[0].transpose(1, 2, 0)
#
#     data_range = norm_ref.max() - norm_ref.min()
#
#     return ssim(norm_img, norm_ref, data_range=1.0, multichannel=True,
#                 win_size=3)