import numpy as np

def add_gaussian_noise(img, noise_level, random_sampling_op):
    np.random.seed(1234)
    gaussian_noise = random_sampling_op(noise_level * np.random.randn(*img.shape))
    return img + gaussian_noise

def apply_poisson_noise(img, alpha):
    np.random.seed(1234)
    val = np.random.poisson(img * alpha)
    return val
