def parse_args_exp (args):
    # Return paresed args related to experimental settings
    # The default value of args should be written here
    # [ gaussian_nl, sp_nl, poisson_noise, poisson_eta, deg_op, r ]
    gaussian_nl     = args['gaussian_nl']      if('gaussian_nl' in args)       else 0
    sp_nl           = args['sp_nl']            if ('sp_nl' in args)            else 0
    poisson_noise   = args['poisson_noise']    if ('poisson_noise' in args)    else False
    poisson_eta     = args['poisson_eta']      if('poisson_eta' in args)       else 1
    deg_op          = args['deg_op']           if ('deg_op' in args)           else 'blur'
    r               = args['r']                if ('r' in args)                else 0.8

    return gaussian_nl, sp_nl, poisson_noise, poisson_eta, deg_op, r

def parse_args_method (args):
    # Return paresed args related to methods and hyper parameters
    # The default value of args should be written here
    # [ method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2 ]
    method          = args['method']           if('method' in args)            else 'ours-A'
    architecture    = args['architecture']     if ('architecture' in args)     else 'DnCNN_nobn_nch_3_nlev_0.01'
    max_iter        = args['max_iter']         if ('max_iter' in args)         else 10
    gamma1          = args['gamma1']           if ('gamma1' in args)           else 1
    gamma2          = args['gamma2']           if('gamma2' in args)            else 1
    alpha_n         = args['alpha_n']          if ('alpha_n' in args)          else 1
    alpha_s         = args['alpha_s']          if ('alpha_s' in args)          else 1
    myLambda        = args['myLambda']         if('myLambda' in args)          else 1
    m1              = args['m1']               if ('m1' in args)               else 15
    m2              = args['m2']               if ('m2' in args)               else 15
    gammaInADMMStep1= args['gammaInADMMStep1'] if ('gammaInADMMStep1' in args) else 0.1
    gammaInADMMStep2= args['lambydaInStep2']   if ('lambydaInStep2' in args)   else 0.1
    
    return method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1, gammaInADMMStep2

def parse_args_configs (args):
    # Return paresed args related to configs
    # The default value of args should be written here
    # [ ch, add_timestamp, result_output ]
    ch              = args['ch']               if('ch' in args)                else 3
    add_timestamp   = args['add_timestamp']    if ('add_timestamp' in args)    else True
    result_output   = args['result_output']    if ('result_output' in args)    else False

    return ch, add_timestamp, result_output