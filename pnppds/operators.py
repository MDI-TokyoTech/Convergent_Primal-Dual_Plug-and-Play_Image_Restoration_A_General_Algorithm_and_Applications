import scipy, scipy.io
import numpy as np

from models.denoiser import Denoiser

def get_blur_operator(x, h):
    # return x*h where * is a convolution
    l = h.shape[0]
    if(np.ndim(x) == 2):
        # grayscale image
        xo = np.pad(x, ((l//2+1, l//2), (l//2+1, l//2)), 'wrap')
        A = np.fft.fft2(h, [xo.shape[-2], xo.shape[-1]])
        y = np.real(np.fft.ifft2(A * np.fft.fft2(xo)))
    elif(np.ndim(x) == 3):
        # color image
        xo = np.pad(x, ((0, 0), (l//2+1, l//2), (l//2+1, l//2)), 'wrap')
        y = np.zeros(xo.shape)
        A = np.fft.fft2(h, [xo.shape[-2], xo.shape[-1]])
        for i in range(xo.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(A * np.fft.fft2(xo[i, ...])))
    return y[..., l:, l:]

def get_adj_blur_operator(x, h):
    l = h.shape[0]
    if(np.ndim(x) == 2):
        # grayscale image
        xo = np.pad(x, ((l//2, l//2), (l//2, l//2)), 'wrap')
        A = np.fft.fft2(h, [xo.shape[-2], xo.shape[-1]])
        y = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo)))
    elif(np.ndim(x) == 3):
        # color image
        xo = np.pad(x, ((0, 0), (l//2, l//2), (l//2, l//2)), 'wrap')
        y = np.zeros(xo.shape)
        A = np.fft.fft2(h, [xo.shape[-2], xo.shape[-1]])
        for i in range(xo.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo[i, ...])))
    return y[..., :-l+1, :-l+1]

def get_random_sampling_operator(x, r):
    w = x.shape[-2]
    h = x.shape[-1]
    degraded_cnt = round(w * h * (1-r))
    q = np.random.RandomState(seed=1234).permutation(w * h)[:degraded_cnt]
    y = np.zeros(x.shape)
    if(np.ndim(x) == 2):
        #grayscale
        t = x.flatten()
        t[q] = 0
        y = t.reshape(w, h)
    elif(np.ndim(x)==3):
        # color
        for i in range(0,3):
            t = np.copy(x[i])
            t = t.flatten()
            t[q] = 0
            y[i] = t.reshape(w, h)
    return y 

def get_observation_operators(operator, path_kernel, r):
    def phi(x):
        if(operator == 'blur'):
            return get_blur_operator(x, h)
        elif (operator == 'random_sampling'):
            return get_random_sampling_operator(x, r)  
        elif (operator == 'Id'):
            return  x
                
    def adj_phi(x):
        if(operator == 'blur'):
            return get_adj_blur_operator(x, h)
        elif (operator == 'random_sampling'):
            return get_random_sampling_operator(x, r)
        elif (operator == 'Id'):
            return  x
    
    h = scipy.io.loadmat(path_kernel)
    h = np.array(h['blur'])
    return phi, adj_phi
    
def denoise(x, path_prox, ch):
    denoiser = Denoiser(file_name=path_prox, ch = ch)
    return denoiser.denoise(x)

def proj_l2_ball(x, alpha_n, gaussian_nl, sp_nl, x_0, r=1):
    # projection on l2 ball
    epsilon = np.sqrt(x.size * (1 - sp_nl)) * r * alpha_n * gaussian_nl
    val = np.copy(x)
    if(np.linalg.norm(x - x_0) > epsilon):
        val = x_0 + epsilon * (x - x_0) / np.linalg.norm(x - x_0)
    return val

def proj_C(x):
    # projection on [0, 1]^N
    val = np.clip(x, 0, 1)
    return val

def prox_GKL(x, gamma, alpha, x_0):
    return 0.5 * (x - gamma * alpha + np.sqrt(np.square(x - gamma * alpha) + 4 * gamma * x_0))