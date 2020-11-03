import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve

def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68*'*')
    print('Gradient Descent')

    # Initialize x and alpha
    #### YOUR CODES HERE
    x=parameter['x0']
    L=parameter['Lips']
    maxit=parameter['maxit']
    str=parameter['strcnvx']
    alpha=1/L
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE
        x_next=x-alpha*gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter %  5 ==0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
    info['iter'] = maxit
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter) :
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """

    print(68*'*')
    print('Gradient Descent  with strong convexity')

    # Initialize x and alpha.
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    alpha = 2/(str+L)



    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE
        x_next=x-alpha*gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                 strcnvx	- strong convexity parameter
    *************************** LIONS@EPFL ***********************************
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient')

    # Initialize x, y and t.

    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    t=1
    y=x
    alpha=1/L



    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE
        x_next=y-alpha*gradf(y)
        t_next=0.5*(1+np.sqrt(1+4*(t**2)))
        y_next=x_next+((t-1)/t_next)*(x_next-x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next
        t = t_next
        y = y_next
    return x, info

# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx	- strong convexity parameter
    *************************** LIONS@EPFL ***********************************
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with strong convexity')

    # Initialize x, y and t.
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    y=x
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE
        x_next=y-(2/(str+L))*gradf(y)
        y_next=x_next+(np.sqrt(L)-np.sqrt(str))/((np.sqrt(L)+np.sqrt(str)))*(x_next-x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next
        y = y_next
    return x, info

# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    print(68 * '*')
    print('Gradient Descent with line search')

    # Initialize x, y and t.
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']


    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE
        i=0
        L_k_0=0.5*L
        while (fx(x +(-gradf(x) / ((L_k_0 * (2 ** i))))) > (
                fx(x) - (np.linalg.norm(-gradf(x))**2) / (L_k_0*(2 ** (i + 1))))):
            i += 1
        L_k = (2 ** i) * L_k_0
        alpha=1/L_k
        x_next=x-alpha*gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        L=L_k



    return x, info

# LSAGD
def LSAGD(fx, gradf, parameter):
    """
        Function:  [x, info] = LSAGD (fx, gradf, parameter)
        Purpose:   Implementation of AGD with line search.
        Parameter: x0         - Initial estimate.
        maxit      - Maximum number of iterations.
        Lips       - Lipschitz constant for gradient.
        strcnvx    - Strong convexity parameter of f(x).
        :param fx:
        :param gradf:
        :param parameter:
        :return:
        """
    print(68 * '*')
    print('Accelerated Gradient with line search')

    # Initialize x, y and t.
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    t = 1
    y = x
    alpha=1/L

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE
        i=0
        L_k_0 = 0.5 * L
        while (fx(y +(-gradf(y) / ((L_k_0 * (2 ** i))))) > (
                fx(y) - (np.linalg.norm(-gradf(y))**2) / (L_k_0*(2 ** (i + 1)) ))):
            i += 1
        L_k = (2 ** i) * L_k_0
        alpha = 1 / L_k

        x_next = y - alpha * gradf(y)
        t_next = 0.5 * (1 + np.sqrt(1 + (4 * (L_k / L) * (t ** 2))))
        y_next= x_next + ((t - 1) / t_next) * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        L = L_k
        y = y_next

    return x, info
# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient with restart')

    # Initialize x, y, t and find the initial function value (fval).
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    t=1;y=x
    alpha=1/L

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE
        x_next=y-alpha*gradf(y)
        if(fx(x)<fx(x_next)):
            t=1;y=x
            x_next = y - alpha * gradf(y)

        t_next = 0.5 * (1 + np.sqrt(1 + (4 * (t ** 2))))
        y_next = x_next + ((t - 1) / t_next) * (x_next - x)



        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        y = y_next

    return x, info




# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient with line search + restart')

    # Initialize x, y, t and find the initial function value (fval).
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    y=x;t=1
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        # LINE SEARCH
        L_k_0 = 0.5 * L
        i=0
        while (fx(y + (-gradf(y) / ((2 ** i) * L_k_0))) >
               (fx(y) - (np.linalg.norm(-gradf(y)) **2/ ((2 ** (i + 1)) * L_k_0)))):
            i = i + 1

        L_k = (2 ** i) * L_k_0
        alpha = 1 / L_k

        # UPDATING THE NEXT ITERATIONS
        x_next = y -alpha*gradf(y)

        # RESTART
        if (fx(x) < fx(x_next)):
            t = 1
            y = x
            x_next = y - alpha * gradf(y)
        t_next = 0.5 * (1 + np.sqrt(1 + (4 * (L_k / L) * (t ** 2))))
        y_next = x_next + ((t - 1) / t_next) * (x_next - x)



        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t=t_next
        L=L_k
        y=y_next

    return x, info

def AdaGrad(fx, gradf, parameter):
    """
    Function:  [x, info] = AdaGrad (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the adaptive gradient method with scalar step-size.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Adaptive Gradient method')
    
    # Initialize x, alpha, delta (and any other)
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    alpha=1
    delta=0.00001
    q=0

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE、
        I=np.identity(len(x))
        q_next=q+np.linalg.norm(gradf(x))
        h=(np.sqrt(q_next)+delta)*I
        A=np.linalg.inv(I)
        x_next=x-alpha*(np.dot(A,gradf(x)))



        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        q=q_next

    return x, info

# Newton
def ADAM(fx, gradf, parameter):
    """
    Function:  [x, info] = ADAM (fx, gradf, hessf, parameter)
    Purpose:   Implementation of ADAM.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param hessf:
    :param parameter:
    :return:
    """

    print(68 * '*')
    print('ADAM')
    
    # Initialize x, beta1, beta2, alphs, epsilon (and any other)
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    alpha=0.1
    beta1=0.9
    beta2=0.999
    epsilon=0.00000001
    m=0
    v=0

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}


    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        g=gradf(x)
        m_next=beta1*m+(1-beta1)*g
        v_next=beta2*v+(1-beta2)*np.square(g)
        m_next_bingo=m_next/(1-beta1**(iter+1))
        v_next_bingo=v_next/(1-beta2**(iter+1))
        h=np.sqrt(v_next_bingo)
        x_next=x-(alpha*m_next_bingo)/(h+epsilon)
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        m=m_next
        v=v_next

    return x, info


def SGD(fx, gradf, parameter):
    """
        Function:  [x, info] = GD(fx, gradf, parameter)
        Purpose:   Implementation of the gradient descent algorithm.
        Parameter: x0         - Initial estimate.
        maxit      - Maximum number of iterations.
        Lips       - Lipschitz constant for gradient.
        strcnvx    - Strong convexity parameter of f(x).
        :param fx:
        :param gradf:
        :param parameter:
        :return:
        """
    print(68 * '*')
    print('Stochastic Gradient Descent')

    # Initialize x and alpha.
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    n = parameter['no0functions']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE
        if(iter<=n):
            alpha_k = 1/(iter+1)
        i_k = np.random.randint(n)
        x_next=x-alpha_k*gradf(x,i_k)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

def SAG(fx, gradf, parameter):
    """
        Function:  [x, info] = GD(fx, gradf, parameter)
        Purpose:   Implementation of the gradient descent algorithm.
        Parameter: x0         - Initial estimate.
        maxit      - Maximum number of iterations.
        Lips       - Lipschitz constant for gradient.
        strcnvx    - Strong convexity parameter of f(x).
        :param fx:
        :param gradf:
        :param parameter:
        :return:
        """
    print(68 * '*')
    print('Stochastic Gradient Descent with averaging')

    # Initialize x and alpha.
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    n = parameter['no0functions']
    Lmax = parameter['Lmax']
    alpha=1/(16*Lmax)
    vec=0
    v_all=0
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE
        i=np.random.randint(n)
        vec=gradf(x,i-1)
        v_all=gradf(x,i)+(n-1)*vec



        x_next=x-(alpha/n)*v_all
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SVR(fx, gradf, gradfsto, parameter):
    """
        Function:  [x, info] = GD(fx, gradf, parameter)
        Purpose:   Implementation of the gradient descent algorithm.
        Parameter: x0         - Initial estimate.
        maxit      - Maximum number of iterations.
        Lips       - Lipschitz constant for gradient.
        strcnvx    - Strong convexity parameter of f(x).
        :param fx:
        :param gradf:
        :param parameter:
        :return:
        """
    print(68 * '*')
    print('Stochastic Gradient Descent with variance reduction')

    # Initialize x and alpha.
    #### YOUR CODES HERE
    x = parameter['x0']
    L = parameter['Lips']
    maxit = parameter['maxit']
    str = parameter['strcnvx']
    n = parameter['no0functions']
    Lmax = parameter['Lmax']
    Q=int(1000*Lmax)
    g=0.01/Lmax

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE

        x_wave=x
        x_all=0
        v_k=gradf(x)
        for k in range(Q):
            i=np.random.randint(n)
            v=gradfsto(x_wave,i)-gradfsto(x,i)+v_k
            x_wave=x_wave-g*v
            x_all=x_all+x_wave


        x_next=(1/(Q))*x_all

 # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info
