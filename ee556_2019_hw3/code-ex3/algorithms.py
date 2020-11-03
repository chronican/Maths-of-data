import time
import numpy as np
from random import randint
from utils import print_end_message, print_start_message, print_progress

def ista(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    tic = time.time()
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + params['lambda'] * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup##############
    x_k = params['x0']
    Lips = params['Lips']
    maxit = params['maxit']

    for k in range(1,maxit+1):

        ############## YOUR CODES HERE ##############
        x_k = proxg(x_k - (1 / Lips) * gradf(x_k), (1 / Lips) * params['lambda'])
        run_details['conv'][k] = fx(x_k) + params['lambda'] * gx(x_k)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(x_k), gx(x_k))


    run_details['X_final'] = x_k

    print_end_message(method_name, time.time() - tic)
    return run_details




def fista(fx, gx, gradf, proxg, params):
    if params['restart_fista']:
        method_name = 'FISTA-RESTART'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)
    tic = time.time()

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + params['lambda'] * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup##############
    x_k = params['x0']
    x_k_1 = params['x0']
    Lips = params['Lips']
    maxit = params['maxit']
    t_k=1
    t_k_1=1


    for k in range(1,maxit+1):
        ############## YOUR CODES HERE##############
        y_k=x_k+t_k*((1/t_k_1)-1)*(x_k-x_k_1)
        x_next=proxg(y_k-(1 / Lips) * gradf(y_k), (1 / Lips)*params['lambda'])
        t_k_1=t_k
        t_k = 0.5*(np.sqrt((t_k_1**4)+4*(t_k_1**2))-t_k_1**2)

        if params['restart_fista'] and gradient_scheme_restart_condition(x_k,x_next,y_k):
            t_k_1=1
            t_k=1
            y_k=x_k
            x_next=proxg(y_k-(1 / Lips) * gradf(y_k), (1 / Lips)*params['lambda'])

        x_k_1=x_k
        x_k=x_next
        # record convergence
        run_details['conv'][k] = fx(x_k) + params['lambda'] * gx(x_k)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(x_k), gx(x_k))

    run_details['X_final'] = x_k

    print_end_message(method_name, time.time() - tic)
    return run_details




def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    ############## YOUR CODES HERE ##############
    a=Y_k-X_k_next
    b=X_k_next-X_k
    if np.trace(np.dot(a.T,b))>0:
        return True
    else:return False


    raise NotImplemented('Implement the method!')





def prox_sg(fx, gx, stocgradfx, proxg, params):
    method_name = 'PROX-SG'
    print_start_message(method_name)

    tic = time.time()


    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + params['lambda']* gx(params['x0'])
    sum1=0
    sum2=0
    minb_size=params['minib_size']
    x=params['x0']
    alpha=params['stoch_rate_regime']
    maxit=params['maxit']
    ############## YOUR CODES HERE - parameter setup##############
    for k in range(1,maxit+1):
            ############## YOUR CODES HERE ##############
        x = proxg(x-alpha(k)*stocgradfx(x,minb_size), alpha(k) * params['lambda'])
        sum1=sum1+alpha(k)
        sum2=sum2+alpha(k)*x
        x_t=sum2/sum1

        run_details['conv'][k] = fx(x_t) +  params['lambda']* gx(x_t)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(x_t), gx(x_t))

    run_details['X_final'] = x############## YOUR CODES HERE ##############
    print_end_message(method_name, time.time() - tic)
    return run_details
