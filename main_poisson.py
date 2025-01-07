import datetime, json
from pnppds.test_pnppds import test_all_images
from utils.utils_textfile import *

with open('config/setup.json', 'r') as f:
    config = json.load(f)

def main():
    experiment_data_list = []
    filepath = config['path_result'] + 'SUMMARY(' + str(datetime.datetime.now().strftime("%Y%m%d %H%M%S %f")) + ').txt'
    touch_textfile (filepath)

    # =====================================
    # Preparation
    # =====================================

    ## Experimental settings
    noise_level_list = [50, 100, 200]
    obs_list         = ['blur', 'random_sampling']

    ## Parameters
    method           = 'Poisson-PnPPDS'
    MAX_ITER_BLUR    = 1200
    MAX_ITER_RS      = 3000
    SAMPLING_RATE    = 0.8
    gamma1           = 0.5
    gamma2           = 0.99
    lambda_bl        = 0.00125
    architecture     = 'DnCNN_nobn_nch_1_nlev_0.01'  # Specify your architecture (e.g. Set 'xxx' for './nn/xxx.pth')
    
    # Configurations
    configs = {'add_timestamp' : False,  'ch' : 1}

    # =====================================
    # Build batches
    # =====================================
    for nl in noise_level_list:
        for obs in obs_list:                
            if (obs == 'blur'):
                max_iter = MAX_ITER_BLUR
                r = 1
            elif (obs == 'random_sampling'):
                max_iter = MAX_ITER_RS
                r = SAMPLING_RATE
            settings =  {'poisson_noise' : True, 'poisson_eta': nl, 'deg_op' : obs, 'r' : r}

            experiment_data_list.append ({'settings' : settings, 'method' : {'method' : method, 'max_iter' : max_iter, 'gamma1' :  gamma1, 'gamma2' :  gamma2, 'myLambda' : lambda_bl, 'architecture' : architecture}, 'configs' : configs})

    # =====================================
    # Run each batch
    # =====================================
    for experiment_data in experiment_data_list:
        data = test_all_images(experiment_data['settings'], experiment_data['method'], experiment_data['configs'])
        write_textfile (filepath, data)

    add_footer_textfile (filepath, data)


if (__name__ == '__main__'):

    main()