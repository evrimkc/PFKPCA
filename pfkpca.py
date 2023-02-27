import numpy as np
import numdifftools as nd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.lines as lines
plt.style.use('bmh')

import time
from tqdm import tqdm

# Utility function: Read the modified MNIST data set
# https://cs.nyu.edu/~roweis/data.html
# ################################################## #
def read_mnist_data(filename, N=1000, pxl=28):
    M = pxl**2
    img = np.zeros(N*M)

    with open(filename, 'rb') as f: ## 'rb':read binary
        data = f.read()

    i = 0
    for byte in data:
        img[i] = byte
        i += 1
    img = img.reshape((N,pxl,pxl))
    
    # Standardize the images, i.e.
    # each image has pixel mean = 0 and std.dev = 1
    for i in range(len(img)):
         img[i] = (img[i] - img[i].mean())/img[i].std()
            
    return img
# ################################################## #


# Kernel matrices
# #################### #
def kernel_matrix_poly(data, coeff, d):
    '''
    Polynomial kernel 2nd degree
    K(x,y) = (1 + x^T . y)^2
    '''
    
    X = data.T
    K = np.array([[(coeff + np.dot(x, y))**d for y in X] for x in X])
    
    return K


def kernel_matrix_rbf(data, rho):
    '''
    RBF Kernel
    K(x, y) = exp(-|| x - y ||^2 / rho)
    '''
    
    X = data.T
    # inter-distances
    norm2 = np.array([[np.dot((x-y).T, x-y) for y in X] for x in X])
    # RBF Kernel
    K = np.exp(-norm2/rho)
    
    return K


# Centering common to both kernels
# ################################
def centered_kernel_matrix(K, unit, unit2, Kone, oneKone):
    '''
    Centering w.r.t. Eq.(9) of A.T. Bui et al. / Neurocomputing 357 (2019) 163–176
    '''
        
    N = K.shape[0]
    K_tilde = K - np.dot(unit2,K)/N - np.dot(K,unit2)/N + np.dot(unit,oneKone).dot(unit.T)/(N**2)

    return K_tilde        
    

# Cost function definitions
# #############################
def cost_poly(z, data, alphas, centering_params, coeff, d):
    '''
    Cost function. Eq.(7) of A.T. Bui et al. / Neurocomputing 357 (2019) 163–176
    || \phi(z) - P \phi(z) ||^2 = K(z,z) - \beta_z^T . \beta_z
    '''
    
    M, N = data.shape
    # Check that z has the correct shape
    assert z.shape == (M,) or z.shape == (M,1),'Dimensions of z {} is wrong. Must be (M,) or (M,1)'.format(z.shape)
    
    # Parameters related to centering
    unit = centering_params['unit']
    unit2 = centering_params['unit2']
    Kone = centering_params['Kone']
    oneKone = centering_params['oneKone']    
    oneKoneUnit2 = centering_params['oneKoneUnit2'] 
    
    # Enforce an (M,1) shape for the data point x
    z = z.reshape((M,1))
    
    # Initialize variables
    Kzz_tilde, BzTBz_tilde = 0.0, 0.0
    
    # K(z,z)
    Kzz = (coeff + np.dot(z.T,z))**d    # (Mx1)^T.(Mx1) = (1x1)
    # K_z
    Kz  = (coeff + np.dot(data.T,z))**d    # (MxN)^T.(Mx1) = (Nx1)
    
    # Centered K(z,z)
    Kzz_tilde = Kzz - 2*np.dot(unit.T, Kz)/N + oneKone/(N**2)  # (1x1)
    
    # Centered K_z and BzTBz
    Kz_tilde = Kz - Kone/N - np.dot(unit2,Kz)/N + oneKoneUnit2/(N**2)  # (Nx1)
    Alphas = np.dot(alphas, alphas.T)  # (NxL).(LxN) = (NxN)
    BzTBz_tilde = (Kz_tilde.T).dot(Alphas).dot(Kz_tilde)  # (1xN).(NxN).(Nx1) = (1x1)
    
    cost = Kzz_tilde - BzTBz_tilde  # (1x1)
    return cost


def cost_rbf(z, data, alphas, rho, centering_params):
    M, N = data.shape
    
    # Check that z has the correct shape
    assert z.shape == (M,) or z.shape == (M,1),'Dimensions of z {} is wrong. Must be (M,) or (M,1)'.format(z.shape)
    
    # Parameters related to centering
    unit = centering_params['unit']
    unit2 = centering_params['unit2']
    Kone = centering_params['Kone']
    oneKone = centering_params['oneKone']    
    oneKoneUnit2 = centering_params['oneKoneUnit2'] 
    
    # Enforce an (M,1) shape for the data point z
    z = z.reshape((M,1))
    
    # Initialize variables
    Kzz_tilde, BzTBz_tilde = 0.0, 0.0
    
    # K(z,z), RBF kernel => e^((z-z)/rho) = 1
    Kzz = 1  
    
    # K_z
    tg = (np.dot(z,unit.T) - data).T  # Distance of point z to each point in the dataset
    Kz = np.exp(-np.sum(tg**2, axis=1)/rho).reshape((N,1)) # enforce (Nx1)
    
    # Centered K(z,z)
    Kzz_tilde = Kzz - 2*np.dot(unit.T, Kz)/N + oneKone/(N**2)


    # Centered K_z and BzTBz
    Kz_tilde = Kz - Kone/N - np.dot(unit2,Kz)/N + oneKoneUnit2/(N**2)  # (Nx1)
    Alphas = np.dot(alphas, alphas.T)  # (NxL).(LxN) = (NxN)
    BzTBz_tilde = (Kz_tilde.T).dot(Alphas).dot(Kz_tilde)  # (1xN).(NxN).(Nx1) = (1x1)
    
    cost = Kzz_tilde - BzTBz_tilde
    return cost


# Gradient function definitions
# #############################
def grad_cost_poly(z, data, alphas, centering_params, coeff, d):
    '''
    Gradient of the cost function w.r.t. z at z = x
    \nabla_z cost|_{z=x}
    '''
    
    d_cost_dz = nd.Gradient(cost_poly)
    grad_cost = d_cost_dz(z, data, alphas, centering_params, coeff, d)    
        
    return grad_cost


def grad_cost_rbf(z, data, alphas, rho, centering_params, num=False):
    '''
    Gradient of the cost function w.r.t. z at z = x
    \nabla_z cost|_{z=x}
    '''
    
    if num:
        d_cost_dz = nd.Gradient(cost_rbf)
        grad_cost = d_cost_dz(z, data, alphas, rho, centering_params)    
    else:
        M, N = data.shape

        # Check that z has the correct shape
        assert z.shape == (M,) or z.shape == (M,1),'Dimensions of z {} is wrong. Must be (M,) or (M,1)'.format(z.shape)

        # Parameters related to centering
        unit = centering_params['unit']
        unit2 = centering_params['unit2']
        ones1n = centering_params['ones1n']
        Kone = centering_params['Kone']
        oneKone = centering_params['oneKone']    
        oneKoneUnit2 = centering_params['oneKoneUnit2'] 

        # Enforce an (M,1) shape for the data point z
        z = z.reshape((M,1))

        # Kz and Gradient of Kz
        norm = (np.dot(z,unit.T) - data).T  # (MxN)
        Kz = np.exp(-np.sum(norm**2, axis=1)/rho).reshape((N,1))  # (Nx1)
        JKz = -2/rho * np.dot(Kz,ones1n) * norm

        Kz_tilde = Kz - Kone/N - np.dot(unit2,Kz)/N + oneKoneUnit2/(N**2)
        JKz_tilde = JKz - np.dot(unit2, JKz)/N
        JBzTBz_tilde = 2*np.dot( np.dot(Kz_tilde.T,alphas), np.dot(alphas.T,JKz_tilde) ).T 

        # Gradient of K(z,z)
        JKzz_tilde = -2 * np.dot(JKz.T,unit)/N
        
        grad_cost = (JKzz_tilde - JBzTBz_tilde).squeeze()
        
    return grad_cost

def kpca(data, L, kernel, coeff, d, rho_m, show_proj, **kwargs):
    '''
    KPCA -- Step 0  pp.168 of A.T. Bui et al. / Neurocomputing 357 (2019) 163–176
    '''
    M, N = data.shape
    
    poly = kernel == 'poly'
    rbf  = kernel == 'rbf'
    # Calculate the kernel (Gram) matrix
    if poly:
        K = kernel_matrix_poly(data, coeff, d)
        rho = None
    elif rbf:
        mean_comp_var = data.var(ddof=1, axis=0).mean()
        rho = (mean_comp_var) * M * rho_m
        K = kernel_matrix_rbf(data, rho)
    else:
        raise ValueError('Unknown kernel')
        
    # Parameters for centering
    unit         = np.ones((N,1))
    unit2        = np.ones((N,N))
    ones1n       = np.ones((1,M))
    Kone         = np.dot(K,unit)
    oneKone      = np.dot(unit.T,Kone)
    oneKoneUnit2 = oneKone*unit
        
    # Centered kernel (Gram) matrix
    K_tilde = centered_kernel_matrix(K, unit, unit2, Kone, oneKone)    
    
    # KPCA
    lambdas, alphas = eigh(K_tilde, eigvals=(N-L,N-1))
    X_kpca = alphas * np.sqrt(lambdas)
    alphas = alphas / np.sqrt(lambdas)

    # Plot the projections to the principal subspaces 
    if show_proj:
        n_cols = 4
        subspace=X_kpca.shape[1]
        if subspace <= n_cols:
            _, ax = plt.subplots(ncols=subspace, figsize=(16,3))
            x_tm = np.linspace(-1.1,1.1)
            for i in range(subspace):
                ax[i].plot(x_tm, x_tm**2, ls='-', marker='', label='true manifold')
                ax[i].plot(data[0,:], data[1,:], ls='', marker='o', markerfacecolor='none', label='data')
                ax[i].plot(data[0,:], X_kpca[:,-i-1], ls='', marker='.', markerfacecolor='none', label='KPCA proj.')
                ax[i].set_xlabel('component #{}'.format(i), fontsize=12, fontstyle='italic')
            ax[0].legend()
        else:
            n_rows = int(np.ceil(subspace/n_cols - 1)) + 1
            fig = plt.figure(figsize=(16,3*n_rows))
            x_tm = np.linspace(-1.1,1.1)
            for i in range(n_rows):
                for j in range(n_cols):
                    idx = j + i*n_cols
                    if idx < subspace:
                        ax = fig.add_subplot(n_rows, n_cols, int(idx+1))
                        ax.plot(x_tm, x_tm**2, ls='-', marker='', label='true manifold')
                        ax.plot(data[0,:], data[1,:], ls='', marker='o', markerfacecolor='none', label='data')
                        ax.plot(data[0,:], X_kpca[:,-idx-1], ls='', marker='.', markerfacecolor='none', label='KPCA proj.')
                        ax.set_xlabel('component #{}'.format(idx), fontsize=12, fontstyle='italic')
            ax.legend()
    
    res = dict(alphas=alphas, rho=rho, 
               centering_params=dict(unit=unit, unit2=unit2, ones1n=ones1n, Kone=Kone, oneKone=oneKone, oneKoneUnit2=oneKoneUnit2)
              )
    return res

def steep_descend_dir(data, alphas, kernel, coeff, d, rho, centering_params, show_gradescent, kwargs):
    '''
    Find the direction of the steepest descend -- Step 1  pp.168 of A.T. Bui et al. / Neurocomputing 357 (2019) 163–176
    '''
    
    M, N = data.shape
    
    poly = kernel == 'poly'
    rbf  = kernel == 'rbf'
    
    if poly:
        descend_dir = -np.array([grad_cost_poly(z, data, alphas, centering_params, coeff, d) for z in tqdm(data.T, desc='PFKPCA: Step 1 - Find the direction of the steepest descend', unit='img', disable=(not kwargs.get('verb', False)))])
    elif rbf:
        descend_dir = -np.array([grad_cost_rbf(z, data, alphas, rho, centering_params, kwargs.get('num',False)) for z in tqdm(data.T, desc='PFKPCA: Step 1 - Find the direction of the steepest descend', unit='img', disable=(not kwargs.get('verb', False)))])
            
    descend_dir = descend_dir/(np.linalg.norm(descend_dir,axis=1).reshape(N,1))  # Unit direction vector (MxN)
    
    # Show a contour plot of the cost and the descend directions if requested
    if show_gradescent:
        # Plot the true manifold
        fig, ax = plt.subplots(constrained_layout=True, figsize=(16,6))

        x_tm = np.linspace(-1.1,1.1)
        ax.plot(x_tm, x_tm**2, color='tab:red', ls='-', marker='')
        ax.set_title('Steepest descend directions', fontsize=22, fontstyle='italic')
        ax.set_xlabel('x', fontsize=18, fontstyle='italic')
        ax.set_ylabel('y', fontsize=18, fontstyle='italic')

        # Calculate the cost on each grid point
        N_grid = kwargs.get('N_grid', 100)
        X, Y = np.meshgrid(np.linspace(-1.2, 1.2, N_grid),np.linspace(-0.5, 1.3, N_grid))
        if poly:
            Z = np.array([[cost_poly(z, data, alphas, centering_params, coeff, d) for z in np.array(mesh_data).T] 
                  for mesh_data in zip(X,Y) ]).reshape(N_grid, N_grid)  
        elif rbf:
            Z = np.array([[cost_rbf(z, data, alphas, rho, centering_params) for z in np.array(mesh_data).T] 
                      for mesh_data in zip(X,Y) ]).reshape(N_grid, N_grid)  

        # Draw contours
        CS = ax.contourf(X, Y, Z, levels=kwargs.get('contour_levels', None))
        # Make a colorbar for the ContourSet returned by the contourf call.
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel('Cost', fontsize=18, fontstyle='italic')

        # Plot the directions
        X, Y = data[0], data[1]  # x and y coordinates of the dataset
        U, V = descend_dir.T[0], descend_dir.T[1]  # Components of the direction vectors
        ax.plot(X, Y, ls='', color='tab:red', marker='o', markerfacecolor='none', label='data')
        Q = ax.quiver(X, Y, U, V, units='xy', angles='xy', color='tab:red', alpha=.5)
        ax.quiverkey(Q, 0.865, 0.17, .9, label=r'$\vec{d}/|\vec{d}|^2$', labelpos='N', coordinates='figure',
                     fontproperties=dict(size=18), labelcolor='tab:red')
        ax.legend(loc='lower right')
    
    return descend_dir

def denoise(data, descend_dir, h, I, alphas, kernel, coeff, d, rho, centering_params, kwargs):
    '''
    Step 2 - Discretize the interval [0,A] for line search  pp.168 of A.T. Bui et al. / Neurocomputing 357 (2019) 163–176
    Step 3 - Denoise the data  pp.168 of A.T. Bui et al. / Neurocomputing 357 (2019) 163–176
    '''
    
    num = kwargs.get('num',False)
    verb = kwargs.get('verb',False)
    
    M, N = data.shape
    
    poly = kernel == 'poly'
    rbf  = kernel == 'rbf'
        
    # Step 2 - Discretize the interval [0,A] for line search
    # ###########################################################
    def step_size(j, A):
        '''
        \alpha_{j+1} - \alpha_j = h^j * A/(I-1), where j = 0, 1, ..., N-2
        Equation given in item S2
        '''
        step = h**j * A/(I-1)
        
        return step            
    # ###########################################################    
    # END Step 2 - Discretize the interval [0,A] for line search
    
    # Step 3 - line search and denoise
    # ###########################################################
    denoised_data = np.zeros((N,M))
    for i in tqdm(range(N), desc='PFKPCA: Step 2 & 3 - Denoise the data', unit='img', disable=(not verb)):
        x_dot_d = data.T.dot(descend_dir[i])
        x_dot_d_i = x_dot_d[i]
        A = np.delete(x_dot_d, i).max() - x_dot_d_i
        
        x = data.T[i]
        coords = np.array([])  # array holding the coordinates along the line search
        cost_x = np.array([])  # array holding the costs at coords. along the line search

        # We need to calculate the cost at the initial coordinates
        # and at the next step  
        j = 0
        x_prev = x + step_size(j, A) * descend_dir[i]  # coordinates of the initial point: x + alpha_0 d = x
        coords = np.append(coords, x_prev)
        
        if poly:
            cost_prev = cost_poly(x_prev, data, alphas, centering_params, coeff, d)  # initial cost
        if rbf:
            cost_prev = cost_rbf(x_prev, data, alphas, rho, centering_params)  # initial cost
        cost_x = np.append(cost_x, cost_prev)  # append to list

        j += 1
        x_next = x_prev + step_size(j, A) * descend_dir[i] # coordinates of the next point
        coords = np.append(coords, x_next)
        
        if poly:
            cost_next = cost_poly(x_next, data, alphas, centering_params, coeff, d)  # cost at the next point
        if rbf:
            cost_next = cost_rbf(x_next, data, alphas, rho, centering_params)  # cost at the next point
        cost_x = np.append(cost_x, cost_next)

        # now we have two costs to compare, we can start the iterative line search
        while cost_prev > cost_next:
            x_prev = x_next
            cost_prev = cost_next

            j += 1
            x_next = x_prev + step_size(j, A) * descend_dir[i] # coordinates of the next point
            coords = np.append(coords, x_next)
            
            if poly:
                cost_next = cost_poly(x_next, data, alphas, centering_params, coeff, d)  # cost at the next point
            if rbf:
                cost_next = cost_rbf(x_next, data, alphas, rho, centering_params)  # cost at the next point
            cost_x = np.append(cost_x, cost_next)
        coords = coords.reshape(-1,M)  # reshape the coordinate array
                
        # Calculate the directional derivative of the cost function at \alpha_{j*-1}
        if poly:
            d_cost = grad_cost_poly(coords[j-1], data, alphas, centering_params, coeff, d).dot(descend_dir[i])
        if rbf:
            d_cost = grad_cost_rbf(coords[j-1], data, alphas, rho, centering_params, num).dot(descend_dir[i])
        
        d_cost = round(d_cost, 2)
        if d_cost < 0:
            denoised_x = (coords[j-1] + coords[j])/2
        elif d_cost == 0:
            denoised_x = coords[j-1]
        else:
            denoised_x = (coords[j-2] + coords[j-1])/2
        denoised_data[i] = denoised_x
    # ###########################################################    
    # END Step 3 - line search and denoise

    # coords[j-1] is z* = alpha_j*-1 in the paper the previous step of the step at which the cost value first started increasing
    return denoised_data.T, coords[j-1] 
    

def visualize_denoising(data, denoised_data, kernel, kwargs):
    '''
    Plot the noisy and denoised datasets
    '''
    
    M, N = data.shape
    
    _, ax = plt.subplots(figsize=(16,6))
    x_tm = np.linspace(-1.1,1.1)
    ax.plot(x_tm, x_tm**2, ls='-', marker='', label='true manifold')
    ax.plot(data[0,:], data[1,:], ls='', marker='o', markerfacecolor='none', label='noisy data')
    ax.plot(denoised_data[0,:], denoised_data[1,:], ls='', marker='o', label='denoised data')

    for i in range(N):
        X = [data[0,i], denoised_data[0,i]]
        Y = [data[1,i], denoised_data[1,i]]
        line = lines.Line2D(X, Y, ls='-', lw=.5, color='black')
        ax.add_line(line)
    
    if kernel == 'poly':
        title = 'PFKPCA denoising with Polynomial Kernel of 2nd degree'
    elif kernel == 'rbf':
        title = 'PFKPCA denoising with RBF Kernel'
    else:
        title = 'PFKPCA denoising'
    ax.set_title(title, fontsize=18, fontstyle='italic')
    ax.set_xlabel('x', fontsize=14, fontstyle='italic')
    ax.set_ylabel('y', fontsize=14, fontstyle='italic')
    ax.legend()
    
    return None


def r1(denoised_data, noisy_data, true_data, verb=False):
    '''
    Assess denoising performance, Eq.(10) of A.T. Bui et al. / Neurocomputing 357 (2019) 163–176
    
    r_1 = \sum_i || \hat{x}_i - x_{i, true} || / \sum_i || x_i - x_{i, true}||,  
    
    where,
        x_i.       : ith noisy data point
        \hat{x}_i  : ith denoised data point 
        x_{i, true}: ith true data point
    '''
    
    r1 = np.linalg.norm(denoised_data - true_data, axis=0).sum() / np.linalg.norm(noisy_data - true_data, axis=0).sum()
    if verb: print("\nPFKPCA: Denoising performance measure \n\t r1 =", r1)
    
    return r1


def pfkpca(data, L, kernel='poly', coeff=1.0, d=2, rho_m=1.0, h=1.2, I=500, 
           show_proj=False, show_gradescent=False, show_denoised=True, **kwargs):
    '''
    Perform a Projection Free Kernel Principal Component Analysis
    A.T. Bui, J.-K. Im and D.W. Apley et al. / Neurocomputing 357 (2019) 163–176
    '''
    t_ini = time.perf_counter()
    
#     if kernel=='rbf':
#         show_proj = False
#         show_gradescent = False
#         show_denoised = False
        
    # Step 0 - Perform KPCA
    # #####################################################################
    if kwargs.get('verb', False): print('PFKPCA: Step 0 - Perform KPCA...')
    
    tic_s0 = time.perf_counter()
    kpca_res = kpca(data, L, kernel, coeff, d, rho_m, show_proj)
    toc_s0 = time.perf_counter()
    
    alphas = kpca_res['alphas']
    rho = kpca_res['rho']
    centering_params = kpca_res['centering_params']
    # #####################################################################    
    # END Step 0 - Perform KPCA
    
    # Step 1 - Direction of the steepest descend
    # #############################################################################################################
#     if kwargs.get('verb', False): print('\nPFKPCA: Step 1 - Direction of the steepest descend...')
        
    tic_s1 = time.perf_counter()
    descend_dir = steep_descend_dir(data, alphas, kernel, coeff, d, rho, centering_params, show_gradescent, kwargs)
    toc_s1 = time.perf_counter()
    # #############################################################################################################
    # END Step 1 - Direction of the steepest descend
    
    # Step 2 & 3 - Denoise the data
    # ###############################################################################################
#     if kwargs.get('verb', False): print('\nPFKPCA: Step 2 & 3 - Denoise the data...')
        
    tic_s23 = time.perf_counter()
    denoised_data, z_star = denoise(data, descend_dir, h, I, alphas, kernel, coeff, d, rho, centering_params, kwargs)
    toc_s23 = time.perf_counter()
    # ###############################################################################################
    # END Step 2 & 3 - Denoise the data
    t_fin = time.perf_counter()
    
    # Visualize the denoised data
    if show_denoised: visualize_denoising(data, denoised_data, kernel, kwargs)
    
    if kwargs.get('verb', False):
        print('\n')
        print('PFKPCA: Step 0 - Perform KPCA took {:0.4f} seconds'.format(toc_s0 - tic_s0))
        print('PFKPCA: Step 1 - Direction of the steepest descend took {:0.4f} seconds'.format(toc_s1 - tic_s1))
        print('PFKPCA: Step 2 & 3 - Denoise the data took {:0.4f} seconds'.format(toc_s23 - tic_s23))
        print('PFKPCA took {:0.4f} seconds'.format(t_fin - t_ini))
    return denoised_data, kpca_res, descend_dir, z_star