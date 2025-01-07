def get_algorithm_denoiser (method):
    # You should add denoiser name and algorithm name for all methods here
    
    method_data = {
        'Gaussian-PnPPDS':       {'algorithm': 'PnP-PDS', 'denoiser' : 'DnCNN'},
        'Poisson-PnPPDS':       {'algorithm': 'PnP-PDS', 'denoiser' : 'DnCNN'},
    }    

    if (method in method_data):
        algorithm = method_data[method]['algorithm']
        denoiser  = method_data[method]['denoiser']
    else :
        algorithm = 'unknown algorithm'
        denoiser  = 'unknown denoiser'

    return algorithm, denoiser