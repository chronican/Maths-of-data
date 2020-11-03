import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import compare_ssim as ssim

from utils import apply_random_mask, psnr, load_image
from operators import TV_norm, Representation_Operator, p_omega, p_omega_t, l1_prox, norm2sq, norm1


# from algorithms import gradient_scheme_restart_condition


def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    ############## YOUR CODES HERE ##############
    a = Y_k - X_k_next
    b = X_k_next - X_k
    if np.trace(np.dot(a.T, b)) > 0:
        return True
    else:
        return False



def ISTA(fx, gx, gradf, proxg, params,F_truth):
    lmbd = params['lambda']

    maxit = params['maxit']
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + params['lambda']*gx(params['x0'])
    fg=np.zeros(maxit+1)
    fg[0] = abs(run_details['conv'][0] - F_truth) / F_truth

    ############## YOUR CODES HERE - parameter setup##############
    x=params['x0']
    Lips=params['Lips']
    tol=params['tol']
    lmbd=params['lambda']

    for k in range(1,maxit+1):

        ############## YOUR CODES HERE ##############
        x=proxg(x-(1/Lips)*gradf(x),1/Lips)
        run_details['conv'][k]=fx(x)+lmbd*gx(x)
        fg[k]=abs(run_details['conv'][k]-F_truth)/F_truth
        if fg[k]<tol:
            break
        #if k % params['iter_print'] == 0:
            #print(k, params['maxit'], run_details['conv'][k], fx(x), gx(x))
        print(run_details['conv'][k])
        print(fg[k])
    #plt.title('F_true_ISTA')
    #plt.xlabel('iteration k')
    #plt.ylabel('log((F(x_k)-F*)-F*)')
    #plt.semilogy(fg)
    #plt.show()

    run_details['X_final'] = x


    return x,run_details['conv'],fg







def FISTA(fx, gx, gradf, proxg, params,F_truth,verbose=False):

    tic = time.time()

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + params['lambda'] * gx(params['x0'])

        ############## YOUR CODES HERE - parameter setup##############
    x_k = params['x0']
    x_k_1 = params['x0']
    Lips = params['Lips']
    maxit = params['maxit']
    t_k = 1
    t_k_1 = 1
    fg = np.zeros(maxit + 1)
    fg[0] = abs(run_details['conv'][0] - F_truth) / F_truth
    tol = params['tol']

    for k in range(1, maxit + 1):
            ############## YOUR CODES HERE##############
        y_k = x_k + t_k * ((1 / t_k_1) - 1) * (x_k - x_k_1)
        x_next = proxg(y_k - (1 / Lips) * gradf(y_k), (1 / Lips))
        t_k_1 = t_k
        t_k = 0.5 * (np.sqrt((t_k_1 ** 4) + 4 * (t_k_1 ** 2)) - t_k_1 ** 2)
        x_k_1 = x_k
        x_k = x_next
        # record convergence
        run_details['conv'][k] = fx(x_k) + params['lambda'] * gx(x_k)
        fg[k] = abs(run_details['conv'][k] - F_truth) / F_truth
        if fg[k] < tol:
            break


        if k % params['iter_print'] == 0:
            print(k, params['maxit'], run_details['conv'][k], fx(x_k), gx(x_k))



    run_details['X_final'] = x_k
    print(time.time() - tic)
    return x_k,run_details['conv'],fg

def FISTA_RESTART(fx, gx, gradf, proxg, params, F_truth,verbose=False):
    tic = time.time()

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + params['lambda'] * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup##############
    x_k = params['x0']
    x_k_1 = params['x0']
    Lips = params['Lips']
    maxit = params['maxit']
    t_k = 1
    t_k_1 = 1
    fg = np.zeros(maxit + 1)
    fg[0] = abs(run_details['conv'][0] - F_truth) / F_truth
    tol = params['tol']

    for k in range(1, maxit + 1):
        ############## YOUR CODES HERE##############
        y_k = x_k + t_k * ((1 / t_k_1) - 1) * (x_k - x_k_1)
        x_next = proxg(y_k - (1 / Lips) * gradf(y_k), (1 / Lips))
        t_k_1 = t_k
        t_k = 0.5 * (np.sqrt((t_k_1 ** 4) + 4 * (t_k_1 ** 2)) - t_k_1 ** 2)
        if  gradient_scheme_restart_condition(x_k, x_next, y_k):
            t_k_1 = 1
            t_k = 1
            y_k = x_k
            x_next = proxg(y_k - (1 / Lips) * gradf(y_k), (1 / Lips))
        x_k_1 = x_k
        x_k = x_next
        # record convergence
        run_details['conv'][k] = fx(x_k) + params['lambda'] * gx(x_k)
        fg[k] = abs(run_details['conv'][k] - F_truth) / F_truth
        if fg[k] < tol:
            break

        if k % params['iter_print'] == 0:
            print(k, params['maxit'], run_details['conv'][k], fx(x_k), gx(x_k))

    run_details['X_final'] = x_k
    print(time.time() - tic)
    return x_k, run_details['conv'], fg



def reconstruct_l1(image, indices, optimizer, params):
    # Wavelet operator
    r = Representation_Operator(m=params["m"])
    m = params["m"]
    N = params['N']
    # Define the overall operator
    # forward_operator = lambda x:r.WT(x)[:,0][indices]
    # p_omega(r.WT(x),indices)
    forward_operator = lambda x: r.WT(x)[indices] ## TO BE FILLED ##  # P_Omega.W^T#
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, m))  ## TO BE FILLED ##  # W. P_Omega^T#
    # adjoint_operator = lambda x: r.W(p_omega_t1(x,indices,N))[:,0]
    # Generate measurements
    b = np.reshape(image, (N, 1), order='F')[indices]
    #b = p_omega(image,indices)## TO BE FILLED ##

    fx = lambda x: norm2sq(b - forward_operator(x))  ## TO BE FILLED ##
    # print(forward_operator)
    gx = lambda x: norm1(x)  # # TO BE FILLED ##
    F_true=fx(r.W(image))+params['lambda']*gx(r.W(image)[indices])
    print(F_true)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: -adjoint_operator(b - forward_operator(x))  ## TO BE FILLED ##
    x, info,fg = optimizer(fx, gx, gradf, proxg, params,F_truth)
    return fg

    #return r.WT(x).reshape((params['m'], params['m']), order="F"), info

def F_true(image,indices,params):
    r = Representation_Operator(m=params["m"])
    N = params['N']
    m = params["m"]
    forward_operator = lambda x: r.WT(x)[indices]  ## TO BE FILLED ##  # P_Omega.W^T#
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, m))
    b = np.reshape(image, (N, 1), order='F')[indices]
    fx = lambda x: norm2sq(b - forward_operator(x))  ## TO BE FILLED #
    gx = lambda x: norm1(x)  # # TO BE FILLED ##
    F_true = fx(r.W(image)) + params['lambda'] * gx(r.W(image))
    print(F_true)
    return F_true
# %%'''

if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################



    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)


    params = {
        'maxit': 1000,
        'tol': 10e-15,
        'Lips': 1,  ## TO BE FILLED ##
        'lambda': 0.01,
        # 'lambda':0.006,## TO BE FILLED ##,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        # 'restart_criterion': ## TO BE FILLED ##, gradient_scheme,#
        # 'stopping_criterion':## TO BE FILLED ##
        'iter_print': 1,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1],
    }
    PATH = './data/gandalf.jpg'  ## TO BE FILLED ##
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, 0.4)
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices
    print(indices)


    #######################################
    # Reconstruction with L1 and TV norms #
    #######################################



    F_truth=F_true(image, indices, params)

    fg1=reconstruct_l1(image, indices, ISTA, params)
    fg2=reconstruct_l1(image, indices, FISTA, params)
    fg3=reconstruct_l1(image, indices, FISTA_RESTART, params)

    plt.title('F_True_convergence')
    plt.xlabel('iteration k')
    plt.ylabel('log((F(x_k)-F_true)-F_true)')
    plt.semilogy(fg1,'r',label='ISTA')
    plt.semilogy(fg2,'b',label='FISTA')
    plt.semilogy(fg3,'g',label='FIsta_restart')
    plt.legend()
    plt.show()




