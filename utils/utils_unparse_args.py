def unparse_args_exp (gaussian_nl, sp_nl, poisson_noise, poisson_eta, deg_op, r):
    # Return unparesed args related to experimental settings
    args = {}
    args['gaussian_nl']     = gaussian_nl
    args['sp_nl']           = sp_nl
    args['poisson_noise']   = poisson_noise
    args['poisson_eta']   = poisson_eta
    args['deg_op']          = deg_op
    args['r']               = r

    return args

def unparse_args_method (method, architecture, max_iter, gamma1, gamma2, alpha_n, alpha_s, myLambda, m1, m2, gammaInADMMStep1, lambydaInStep2):
    # Return unparesed args related to methods and hyper parameters
    args = {}
    args['method']          = method
    args['architecture']    = architecture
    args['max_iter']        = max_iter
    args['gamma1']          = gamma1
    args['gamma2']          = gamma2
    args['alpha_n']         = alpha_n
    args['alpha_s']         = alpha_s
    args['myLambda']        = myLambda
    args['m1']              = m1
    args['m2']              = m2
    args['gammaInADMMStep1']= gammaInADMMStep1
    args['lambydaInStep2']  = lambydaInStep2
    
    return args

def unparse_args_configs (ch, add_timestamp, result_output):
    # Return unparesed args related to configs
    args = {}
    args['ch']              = ch
    args['add_timestamp']   = add_timestamp
    args['result_output']   = result_output
    
    return args