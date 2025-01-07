import numpy as np
from skimage.metrics import structural_similarity as ssim

def eval_psnr(im1, im2):
    data_range=1
    mse = np.mean((im1.astype(float) - im2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)

def eval_ssim(im1, im2):
    data_range = 1 
    ssim_val = ssim(im1 = im1, im2 = im2, data_range = data_range, channel_axis = 0)
    return ssim_val