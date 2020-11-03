import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import compare_ssim as ssim

from utils import apply_random_mask, psnr, load_image
from operators import TV_norm, Representation_Operator, p_omega, p_omega_t, l1_prox,norm2sq,norm1
#from algorithms import gradient_scheme_restart_condition
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    ############## YOUR CODES HERE ##############
    a=Y_k-X_k_next
    b=X_k_next-X_k
    if np.trace(np.dot(a.T,b))>0:
        return True
    else:return False


def FISTA_RESTART(fx, gx, gradf, proxg, params):
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
    #forward_operator = lambda x:r.WT(x)[:,0][indices]
        #p_omega(r.WT(x),indices)
    forward_operator = lambda x:r.WT(x)[indices]## TO BE FILLED ##  # P_Omega.W^T#
    adjoint_operator = lambda x:r.W(p_omega_t(x,indices,m))## TO BE FILLED ##  # W. P_Omega^T#
    b=np.reshape(image,(N,1),order='F')[indices]
    #b = p_omega(image,indices)## TO BE FILLED ##

    fx = lambda x:norm2sq(b - forward_operator(x))## TO BE FILLED ##
    #print(forward_operator)
    gx = lambda x:norm1(x)## TO BE FILLED ##
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)

    gradf = lambda x:-adjoint_operator(b - forward_operator(x)) ## TO BE FILLED ##

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







    #######################################
    # Reconstruction with L1 and TV norms #
    #######################################
if __name__ == "__main__":
        ##############################
        # Load image and sample mask #
        ##############################
    shape = (256, 256)

    params = {
            'maxit': 500,
            'tol': 10e-15,
            'Lips': 1,  ## TO BE FILLED ##
            'lambda': 0.01,

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

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices
    print(indices)

        #######################################
        # Reconstruction with L1 and TV norms #
        #######################################
    t_start = time.time()
    print("xdc")
    reconstruction_l1 = reconstruct_l1(image, indices, FISTA_RESTART, params)[0]
    t_l1 = time.time() - t_start

    psnr_l1 = psnr(image, reconstruction_l1)
    ssim_l1 = ssim(image, reconstruction_l1)

    t_start = time.time()
    reconstruction_tv = reconstruct_TV(image, indices, FISTA_RESTART, params)[0]
    t_tv = time.time() - t_start

    psnr_tv = psnr(image, reconstruction_tv)
    ssim_tv = ssim(image, reconstruction_tv)

    # Plot the reconstructed image alongside the original image and PSNR
    '''
    fig1, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    ax[2].imshow(reconstruction_l1, cmap="gray")
    ax[2].set_title('\nL1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
    ax[3].imshow(reconstruction_tv, cmap="gray")  #
    ax[3].set_title('\nTV - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.show()
    
    '''

    fig1, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    ax[2].imshow(reconstruction_l1, cmap="gray")
    ax[2].set_title('\nL1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
    #ax[3].imshow(reconstruction_tv, cmap="gray")  #
    #ax[3].set_title('\nTV - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
    im = ax[3].imshow(abs(image -reconstruction_l1 ), cmap="inferno", vmax=.05)
    print('error_l1=',abs(image -reconstruction_l1).sum())
    # Plot the colorbar
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig1.colorbar(im, cax=cax, orientation='vertical');
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.show()


    fig2, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    #ax[2].imshow(reconstruction_l1, cmap="gray")
    #ax[2].set_title('\nL1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
    ax[2].imshow(reconstruction_tv, cmap="gray")  #
    ax[2].set_title('\nTV - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
    im = ax[3].imshow(abs(image -reconstruction_l1 ), cmap="inferno", vmax=.05)
    print('error_tv=', abs(image - reconstruction_l1).sum())
    # Plot the colorbar
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig2.colorbar(im, cax=cax, orientation='vertical');
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.show()

