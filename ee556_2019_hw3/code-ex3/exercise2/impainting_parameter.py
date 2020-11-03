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


def FISTA(fx, gx, gradf, proxg, params):
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

    for k in range(1, maxit + 1):
        ############## YOUR CODES HERE##############
        y_k = x_k + t_k * ((1 / t_k_1) - 1) * (x_k - x_k_1)
        x_next = proxg(y_k - (1 / Lips) * gradf(y_k), (1 / Lips) )
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

        if k % params['iter_print'] == 0:
            print(k, params['maxit'], run_details['conv'][k], fx(x_k), gx(x_k))

    run_details['X_final'] = x_k

    print(time.time() - tic)

    return x_k, run_details['conv']






def reconstruct_l1(image, indices, optimizer, params):
    # Wavelet operator
    r = Representation_Operator(m=params["m"])
    m = params["m"]
    N = params['N']
    # Define the overall operator
    # forward_operator = lambda x:r.WT(x)[:,0][indices]
    # p_omega(r.WT(x),indices)
    forward_operator = lambda x: r.WT(x)[indices]  ## TO BE FILLED ##  # P_Omega.W^T#
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, m))  ## TO BE FILLED ##  # W. P_Omega^T#
    # adjoint_operator = lambda x: r.W(p_omega_t1(x,indices,N))[:,0]
    # Generate measurements
    b = np.reshape(image, (N, 1), order='F')[indices]
    # b = p_omega(image,indices)## TO BE FILLED ##

    fx = lambda x: norm2sq(b - forward_operator(x))  ## TO BE FILLED ##
    # print(forward_operator)
    gx = lambda x: norm1(x)  ## TO BE FILLED ##
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)

    gradf = lambda x: -adjoint_operator(b - forward_operator(x))  ## TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return r.WT(x).reshape((params['m'], params['m']), order="F"), info

def reconstruct_TV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    m = params["m"]
    N = params['N']
    forward_operator = lambda x:p_omega(x,indices)## TO BE FILLED ##  # P_Omega.W^T
    adjoint_operator = lambda x:p_omega_t(x,indices,m)## TO BE FILLED ##  # W. P_Omega^T

    # Generate measurements
    b = np.reshape(image, (N, 1), order='F')[indices]## TO BE FILLED ##

    fx = lambda x:norm2sq(b - forward_operator(x)) ## TO BE FILLED ##
    gx = lambda x: TV_norm(x)## TO BE FILLED ##
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m']), order="F"),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1), order="F")
    gradf = lambda x: -adjoint_operator(b - forward_operator(x)).reshape(-1,1)## TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)

    return x.reshape((params['m'], params['m']), order="F"), info
if __name__ == "__main__":
        ##############################
        # Load image and sample mask #
        ##############################
    shape=(256, 256)
    itera = np.linspace(0.0001, 0.1, 20)
    y1 = np.zeros(20)
    y2 = np.zeros(20)
    PATH = './data/gandalf.jpg'  ## TO BE FILLED ##
    image = load_image(PATH, shape)

    im_us, mask = apply_random_mask(image, 0.4)
    indices = np.nonzero(mask.flatten(order='F'))[0]
    #params['indices'] = indices
    print(indices)


    for i in range(20):
        params = {
                'maxit': 200,
                'tol': 10e-15,
                'Lips': 1,  ## TO BE FILLED ##
                'lambda': itera[i],  ## TO BE FILLED ##,
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


        reconstruction_l1 = reconstruct_l1(image, indices, FISTA, params)[0]
        y1[i] = psnr(image, reconstruction_l1)
        reconstruction_tv = reconstruct_TV(image, indices, FISTA, params)[0]
        y2[i] = psnr(image, reconstruction_tv)

    plt.title('lambda sweep')
    plt.xlabel('lambda')
    plt.ylabel('psnr')
    plt.xscale('log')
    plt.plot(itera, y1, 'r', label='L1-norm')
    plt.plot(itera, y2, 'b', label='TV-norm')
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()