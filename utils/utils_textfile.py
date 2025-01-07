def get_csv_header ():
    myStr = ''
    myStr += 'Observation,'
    myStr += 'Gaussian_noise,'
    myStr += 'poisson_eta,'
    myStr += 'method,'
    myStr += 'algorithm,'
    myStr += 'denoiser,'
    myStr += 'PSNR,'
    myStr += 'SSIM,'
    myStr += 'AverageTime,'
    myStr += 'gamma1,'
    myStr += 'gamma2,'
    myStr += 'alpha_n,'
    myStr += 'myLambda,'
    myStr += 'max_iter,'
    myStr += 'm1,'
    myStr += 'm2,'
    myStr += 'r,'
    myStr += 'ch,'
    myStr += 'Result PSNR - Result SSIM - Observed PSNR - Observed SSIM (for each images) - CPU Time\n'
    return myStr

def get_csv_data (data):
    myStr = ''
    myStr += str(data['experimental_settings']['deg_op']) + ','
    myStr += str(data['experimental_settings']['gaussian_nl']) + ','
    myStr += str(data['experimental_settings']['poisson_eta']) + ','
    myStr += str(data['method']['method']) + ','
    myStr += str(data['summary']['algorithm']) + ','
    myStr += str(data['summary']['denoiser']) + ','
    myStr += str(data['summary']['Average_PSNR']) + ','
    myStr += str(data['summary']['Average_SSIM']) + ','
    myStr += str(data['summary']['Average_time']) + ','
    myStr += str(data['method']['gamma1']) + ','
    myStr += str(data['method']['gamma2']) + ','
    myStr += str(data['method']['alpha_n']) + ','
    myStr += str(data['method']['myLambda']) + ','
    myStr += str(data['method']['max_iter']) + ','
    myStr += str(data['method']['m1']) + ','
    myStr += str(data['method']['m2']) + ','
    myStr += str(data['experimental_settings']['r']) + ','
    myStr += str(data['configs']['ch']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['PSNR']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['SSIM']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['PSNR_observation']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['SSIM_observation']) + ','
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['CPU_time']) + ','
    return myStr

def get_csv_footer (data):
    myStr = ''
    for result_for_single_image in data['results'].values():
        myStr += str(result_for_single_image['filename']) + ','
    return myStr

def touch_textfile (filepath):
    f = open(filepath, 'w')
    f.write(get_csv_header())
    f.close()
    return

def write_textfile (filepath, data):
    f = open(filepath, 'a')
    f.write(get_csv_data(data) + '\n')
    f.close()
    return

def add_footer_textfile (filepath, data):
    f = open(filepath, 'a')
    f.write(get_csv_footer(data) + '\n')
    f.close()
    return