import numpy as np
import pnppds.operators as op
import time, torch

from models.denoiser import Denoiser as Denoiser_J
from utils.utils_eval import eval_psnr, eval_ssim

def test_iter(x_0, x_obsrv, x_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2, gaussian_nl, sp_nl, poisson_eta, path_prox, max_iter, method="Gaussian-PnPPDS", ch = 3, r=1):
    # x_0             Initial value
    # x_obsrv         Observed image
    # x_true          Ground-truth image
    # phi, adj_phi    Observation operator and its adjoint operator
    # gamma1, gamma2  Step sizes of PDS
    # alpha_n         Upper bound coefficient in constraint formulations
    # myLambda        Balancing parameter in additive formulations
    # gaussian_nl     Standard deviation of Gaussian noise
    # path_prox       Path to the Gaussian denoiser
    # max_iter        Number of iterations

    x_n = x_0
    y_n = np.zeros(x_0.shape) # 次元が画像と同じ双対変数
    y2_n = np.zeros(x_0.shape)
    s_n = np.zeros(x_0.shape)
    c = np.zeros(max_iter)
    psnr_data = np.zeros(max_iter)
    ssim_data = np.zeros(max_iter)
    evol_data = np.zeros(x_0.shape)

    denoiser_J = Denoiser_J(file_name=path_prox, ch = ch)

    start_time = time.process_time()
    for i in range(max_iter):
        x_prev = x_n

        if(method == 'Gaussian-PnPPDS'):
            # Plug-and-play Primal-dual spilitting algorithm with FNE DnCNN (Gaussian noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.proj_l2_ball(y_n / gamma2, alpha_n, gaussian_nl, sp_nl, x_obsrv, r)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)

        elif(method == 'Poisson-PnPPDS'):
            # Plug-and-play Primal-dual spilitting algorithm with FNE DnCNN (Poisson noise)
            x_n = denoiser_J.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
            y_n = y_n + gamma2 * phi(2 * x_n - x_prev)
            y2_n = y2_n + gamma2 * (2 * x_n - x_prev)
            y_n = y_n - gamma2 * op.prox_GKL(y_n / gamma2, myLambda / gamma2, poisson_eta, x_obsrv)
            y2_n = y2_n - gamma2 * op.proj_C(y2_n / gamma2)

        else:
            print("Unknown method:", method)
            return x_n, s_n+0.5, c, psnr_data, ssim_data, average_time

        c[i] = np.linalg.norm((x_n - x_prev).flatten(), 2) / np.linalg.norm(x_prev.flatten(), 2)
        psnr_data[i] = eval_psnr(x_true, x_n)
        ssim_data[i] = eval_ssim(x_true, x_n)

    torch.cuda.synchronize(); 
    end_time = time.process_time()
    average_time = (end_time - start_time)/max_iter

    others_data = {}
    others_data['evol_data'] = evol_data

    return x_n, s_n+0.5, c, psnr_data, ssim_data, average_time, others_data

