import cv2, datetime, glob, json, os, pnppds.iteration as iteration
import numpy as np
from pnppds.operators import get_observation_operators
from utils.utils_image import save_imgs
from utils.utils_noise import *
from utils.utils_eval import eval_psnr, eval_ssim
from utils.utils_parse_args import *
from utils.utils_unparse_args import *
from utils.utils_method_master import get_algorithm_denoiser
from utils.utils_textfile import *

with open('config/setup.json', 'r') as f:
    config = json.load(f)

def test_all_images (experimental_settings_arg = {}, method_arg = {}, configs_arg = {}):
    gaussian_nl, sp_nl, poisson_noise, poisson_eta, deg_op, r  = parse_args_exp (experimental_settings_arg)
    method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2 = parse_args_method (method_arg)
    ch, add_timestamp, result_output = parse_args_configs (configs_arg)
    experimental_settings_all = unparse_args_exp (gaussian_nl, sp_nl, poisson_noise, poisson_eta, deg_op, r)
    method_all = unparse_args_method (method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2)
    configs_all = unparse_args_configs (ch, add_timestamp, result_output)

    file_ext = config['file_ext']
    path_test   = config['path_test']
    path_result = config['path_result']
    path_kernel = config['root_folder'] + 'blur_models/' + 'blur_1' + '.mat'
    path_prox   = config['root_folder'] + 'nn/' + architecture + '.pth'
    path_images = sorted(glob.glob(os.path.join(path_test, file_ext)))

    psnr = np.zeros((len(path_images)))
    ssim = np.zeros((len(path_images)))
    cpu_time = np.zeros((len(path_images)))
    results = {}

    for path_img in path_images:
        # =====================================
        # Prepare images and operators
        # =====================================
        index = path_images.index(path_img)
        img_true = cv2.imread(path_img)
        img_true = np.asarray(img_true, dtype="float32")/255.
        if(ch == 1):
            # Gray scale
            img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        elif(ch == 3):
            # Color  (3 x H x W)
            img_true = np.moveaxis(img_true, -1, 0)
            
        phi, adj_phi = get_observation_operators(operator = deg_op, path_kernel = path_kernel, r = r)
        Id, _ = get_observation_operators("Id", path_kernel, r)
        img_obsrv = phi(img_true)
        if(deg_op == 'blur' or deg_op == 'Id'):
            img_obsrv = add_gaussian_noise(img_obsrv, gaussian_nl, Id)
        elif (deg_op == 'random_sampling'):
            img_obsrv = add_gaussian_noise(img_obsrv, gaussian_nl, phi)        
        if(poisson_noise):
            img_obsrv = apply_poisson_noise(img_obsrv, poisson_eta)
        x_0 = np.copy(img_obsrv)
        if(poisson_noise):
            x_0 = x_0 / poisson_eta
        
        # =====================================
        # Run evaluation
        # =====================================
        img_sol, s_sol, c_evolution, psnr_evolution, ssim_evolution, average_time, other_data = iteration.test_iter(x_0, img_obsrv, img_true, phi, adj_phi, gamma1, gamma2, alpha_s, alpha_n, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2, gaussian_nl, sp_nl, poisson_eta, path_prox, max_iter, method, ch, r)
        if(poisson_noise):
            img_obsrv = img_obsrv / poisson_eta

        # =====================================
        # Save results
        # =====================================
        filename = (path_img[path_img.rfind('\\'):])[1:]
        psnr[index] = psnr_evolution[-1]
        ssim[index] = ssim_evolution[-1]
        cpu_time[index] = average_time
        psnr_obsrv = eval_psnr(img_true, img_obsrv)
        ssim_obsrv = eval_ssim(img_true, img_obsrv)
        results[index] = {'filename' : filename, 'c_evolution': c_evolution, 'PSNR_evolution' : psnr_evolution, 'SSIM_evolution' : ssim_evolution, 'GROUND_TRUTH': img_true, 'OBSERVATION' : img_obsrv, 'RESULT': img_sol, 'REMOVED_SPARSE': s_sol, 'PSNR' : psnr_evolution[-1], 'SSIM' : ssim_evolution[-1], 'CPU_time' : average_time, 'PSNR_observation' : psnr_obsrv, 'SSIM_observation' : ssim_obsrv}
        results[index]['other_data'] = other_data

        # =====================================
        # Save images
        # =====================================
        pictures = [img_true, img_obsrv, img_sol]
        path_saveimg_base = method + '_' + deg_op + '_' + str(gaussian_nl).ljust(5, '0') + '_' + str(poisson_eta) + '_(' + filename + ')' + '_alpha' + str(alpha_n).ljust(5, '0') + '_lambda' + str(myLambda).ljust(5, '0')
        if (add_timestamp):
            path_saveimg_base = path_saveimg_base + '_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        path_pictures = [path_result + '\GROUND_TRUTH_' + path_saveimg_base,  path_result + '\OBSERVATION_' + path_saveimg_base, path_result + '\RESULT_' + path_saveimg_base]
        save_imgs(pictures = pictures, path_pictures = path_pictures, format = '.png')
        
        timestamp_commandline = str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print(timestamp_commandline + '  (' + str(index+1) + '/' + str(len(path_images)) + ') PSNR:' + str(psnr[index].round(3)).ljust(6, '0') + '    SSIM:' + str(ssim[index].round(3)).ljust(6, '0') +  '   ' + filename)


    # =====================================
    # Save all results
    # =====================================
    timestamp_commandline = str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    algorithm, denoiser = get_algorithm_denoiser (method)
    summary = {'Average_PSNR':np.mean(psnr), 'PSNR':psnr, 'Average_SSIM':np.mean(ssim), 'SSIM' : ssim, 'Average_time':np.average(cpu_time) , 'Cpu_time': cpu_time, 'algorithm' : algorithm, 'denoiser' : denoiser}
    datas = {'experimental_settings' : experimental_settings_all, 'method' : method_all, 'configs' : configs_all, 'results' : results, 'summary' : summary}
    np.save(path_result + '\DATA_' + path_saveimg_base, datas)

    print(timestamp_commandline + '  Average_PSNR:' + str(np.mean(psnr).round(3)) + '  Average_SSIM:' + str(np.mean(ssim).round(3)) + '    Algorithm:' +  method + '   Observation:' + deg_op + '   Gaussian noise level:' + str(gaussian_nl).ljust(5, '0'))

    return datas