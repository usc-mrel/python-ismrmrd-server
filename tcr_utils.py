
import numpy as np
import cupy as cp
import time
import logging
import cupyx.scipy.ndimage as cpsp


def extract_middle_slices(array, n, dim, shift=0):
    """
    Extract the middle `n` slices from a 3D NumPy array along a specified dimension.

    Parameters:
    array (np.ndarray): The 3D NumPy array from which slices will be extracted.
    n (int): The number of middle slices to extract.
    dim (int): The dimension along which to extract slices (0, 1, or 2).

    Returns:
    np.ndarray: A 4D NumPy array containing the middle `n` slices along the specified dimension.
    """
    if dim not in [0, 1, 2]:
        raise ValueError("Dimension must be 0, 1, or 2.")
    
    if n <= 0:
        raise ValueError("Number of slices `n` must be greater than 0.")
    
    # Get the size of the specified dimension
    shape = array.shape
    size = shape[dim]
    
    if size < n:
        raise ValueError("The array dimension size is smaller than the number of slices requested.")
    
    # Calculate the start and end indices for slicing
    mid_index = size // 2
    mid_index = mid_index + shift 
    half_n = n // 2
    
    start = max(mid_index - half_n, 0)
    end = min(mid_index + half_n + (n % 2), size)
    
    if dim == 0:
        return array[start:end, :, :]
    elif dim == 1:
        return array[:, start:end, :]
    elif dim == 2:
        return array[:, :, start:end]

def crop_middle(data, dims, perc):
    # function to crop data along axes specified in dims, perc = percentage of slices to keep
    n_slices_keep = [int(np.floor(data.shape[i] * perc)) for i in range(len(data.shape))]
    middle_slice = [int(np.floor(data.shape[i] / 2)) + 1 for i in range(len(data.shape))]
    bounds = [(int(middle_slice[i] - np.floor(n_slices_keep[i]/2)), int(middle_slice[i] + np.floor(n_slices_keep[i]/2))) for i in range(len(data.shape))]
    slices = [slice(bounds[i][0], bounds[i][1]) if i in dims else slice(None) for i in range(len(bounds))]
    dat_out =  data[tuple(slices)]

    return dat_out

"""
MATLAB CODE
function[Q,R] = grahmschmidt(A)
% take matrix [m,n] and solve grahm schidt.
% taken from mit course 18.06
[m,n] = size(A);
Q = zeros(m,n);
R = zeros(m,n);
for j = 1:n
    v = A(:,j);
    for i = 1:(j-1)
        R(i,j) = Q(:,i)'*A(:,j);
        v = v-R(i,j)*Q(:,i);
    end
    R(j,j) = norm(v);
    Q(:,j) = v/R(j,j);
end
end
"""

def gram_schmidt(A):
    [m,n] = A.shape
    Q = cp.zeros((m,n), dtype=cp.complex64)
    R = cp.zeros((m,n), dtype=cp.complex64)
    for j in range(n):
        v = A[:,j]
        for i in range(j-1):
            R[i,j] = cp.dot(Q[:,i].conj().T, A[:,j])
            v = v - R[i,j]*Q[:,i]
        R[j,j] = cp.linalg.norm(v)
        Q[:,j] = v/R[j,j]
    return Q

def rovir_apply(data, W, n_coil_keep):
    W = cp.copy(W)
    n_coil = W.shape[0]
    W = W[:, 0:n_coil_keep]
    data_size = data.shape

    W = cp.copy(gram_schmidt(W))
    #[Q, R] = np.linalg.qr(W)
    #W = np.copy(Q)

    # ensure data.shape[-1] == n_coil 
    assert data.shape[-1] == n_coil

    data = data.reshape([np.prod(data_size[0:-1]),n_coil])
    data = data @ W
    data = data.reshape(list(data_size[0:-1]) + [n_coil_keep])
    return data

def online_tcr_ISTA(A, xn_1, ATyn, lam, max_eig, n_iter=40, del_0=None, cost_fn=None):
    costs = []
    if del_0 is None:
        del_n = cp.zeros(xn_1.shape)
    else:
        del_n = cp.copy(del_0)

    for i in range(n_iter):
        del_n = soft_threshold(del_n - (max_eig*A.N*(del_n+xn_1)) + (max_eig*ATyn), lam)
        del_n = crop_FOV_edge(del_n, 0.8)

        if cost_fn is not None:
            costs.append(cost_fn(del_n+xn_1) + (lam * cp.abs(cp.sum(del_n))))
            
    if cost_fn is not None:
        return [del_n, costs]

    return del_n

# online TCR reconstruction
def online_tcr_POGM(A, xn_1, ATyn, lam, step, n_iter=40, del_0=None, cost_fn=None):
    # understanding variable mapping <->
    # xk = del_n
    # yk = new variable
    # tk = new variable
    costs = []
    if del_0 is None:
        del_n = cp.zeros(xn_1.shape)
    else:
        del_n = cp.copy(del_0)
    
    w_n_1 = del_n
    theta_n_1 = 1
    
    z_n_1 = del_n 
    del_n_1 = del_n
    
    gamma_n_1 = 1

    for i in range(n_iter):
        if (i+1) < n_iter:
            theta_n = 0.5 * (1 + cp.sqrt((4 * cp.square(theta_n_1)) + 1))
        else:
            theta_n = 0.5 * (1 + cp.sqrt((8 * cp.square(theta_n_1)) + 1)) 

        gamma_n = step * (((2 * theta_n_1) + theta_n - 1) / (theta_n))
        w_n = del_n_1 - (step*A.N*(del_n_1+xn_1)) + (step*ATyn)

        z_n = w_n + (((theta_n_1 - 1) / (theta_n)) * (w_n - w_n_1)) + (((theta_n_1) / (theta_n)) * (w_n - del_n_1)) + \
            (((theta_n_1 - 1) / ((1/step)*gamma_n_1*theta_n)) * (z_n_1 - del_n_1))
        del_n = soft_threshold(z_n, lam)
        del_n = crop_FOV_edge(del_n, 0.8)

        # update theta, del, gamma, w, and z.
        theta_n_1 = theta_n
        del_n_1 = del_n
        gamma_n_1 = gamma_n
        w_n_1 = w_n
        z_n_1 = z_n

        if cost_fn is not None:
            costs.append(cost_fn(del_n+xn_1) + (lam * cp.abs(cp.sum(del_n))))

    if cost_fn is not None:
        return [del_n, costs]
    return del_n

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = cp.ogrid[:h, :w]
    dist_from_center = cp.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


# crop edges of the FOV
def crop_FOV_edge(x, amt=1):
    if amt > 1:
        print("AMT NEEDS TO BE LESS THAN 1")
        amt = 1
    x_fft = cp.fft.fftshift(cp.fft.fft2(x, axes=[0,1]), axes=[0,1])
    mask = create_circular_mask(x.shape[0], x.shape[1], radius=(x.shape[0]*amt/2))
    mask = cp.expand_dims(mask, 2)
    x_fft = cp.multiply(x_fft , mask)
    x_recomp = cp.fft.ifft2(cp.fft.ifftshift(x_fft, axes=[0,1]), axes=[0,1])
    return x_recomp

def norm(x):
    # L2 norm of vectorized x.
    # flatten x
    norm = (cp.sum(cp.square(cp.abs(x.flatten()))))
    return norm


def cg(Af, b, x0, niter=20, tol=1e-6):
    x = cp.copy(x0)
    r = Af(x) - b
    p = cp.copy(r)
    for i in range(niter):
        print(f"iter {i}")
        Ap = Af(p)
        rsold = cp.sum(cp.square(cp.abs(r)))
        alpha = rsold / np.vdot(p, Ap) 
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = cp.sum(cp.square(cp.abs(r)))
        if rsnew < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def FISTA_iteration(iter, xk_1, gradf, step_size, fista_dict=None, threshold=0, max_iter=10):
    if iter == 0:
        fista_dict = {
            'yk_1': xk_1,
            'tk_1': 1
        }
    elif iter >= max_iter:
        print("not iterating, max_iter reached")
        return [xk_1, fista_dict]
    tk_1 = fista_dict['tk_1']
    yk_1 = fista_dict['yk_1']

    tk = 0.5 * (1 + cp.sqrt((4 * cp.square(tk_1)) + 1))
    xk = yk_1 - (step_size * gradf(yk_1))
    xk = soft_threshold(xk, threshold)
    yk = xk + (((tk_1 - 1) / tk) * (xk - xk_1))
    return [xk, {'yk_1': yk, 'tk_1': tk}]

def POGM_iteration(iter, xk_1, gradf, step_size, pogm_dict=None, threshold=0, max_iter=10):
    if iter == 0:
        pogm_dict = {
            'zk_1': xk_1,
            'thetak_1': 1,
            'wk_1': xk_1,
            'gammak_1': 1
        }
    elif iter >= max_iter:
        print("not iterating, max_iter reached")
        return [xk_1, pogm_dict]
    thetak_1 = pogm_dict['thetak_1']
    gammak_1 = pogm_dict['gammak_1']
    zk_1 = pogm_dict['zk_1']
    wk_1 = pogm_dict['wk_1']

    if iter < max_iter:
        thetak = 0.5 * (1 + cp.sqrt((4 * cp.square(thetak_1)) + 1))
    else:
        thetak = 0.5 * (1 + cp.sqrt((8 * cp.square(thetak_1)) + 1))

    gammak = step_size * (((2 * thetak_1) + thetak - 1) / (thetak))
    wk = xk_1 - (step_size * gradf(xk_1))

    zk = wk + (((thetak_1 - 1) / thetak) * (wk - wk_1)) + (((thetak_1) / thetak) * (wk - xk_1)) + \
            + (((thetak_1 - 1) * step_size / (gammak_1 * thetak)) * (zk_1 - xk_1))
    
    xk = soft_threshold(zk, threshold)
    return [xk, {'zk_1': zk, 'thetak_1': thetak, 'wk_1': wk, 'gammak_1': gammak}]

def online_TCR_POGM_2(A, xn_1, ATyn, lam, step, n_iter=40, del_0=None, cost_fn=None):

    # construct gradf function
    gradf = lambda x: A.N * (x + xn_1) - ATyn

    if del_0 is None:
        del_n = cp.zeros(xn_1.shape)
    else:
        del_n = cp.copy(del_0)

    #initialize variables
    pogm_dict = None
    costs = []

    for i in range(n_iter):
        [del_n, pogm_dict] = POGM_iteration(i, del_n, gradf, step, pogm_dict, lam*step, n_iter)
        del_n = crop_FOV_edge(del_n, 0.6)
        if cost_fn is not None:
            costs.append(cost_fn(del_n+xn_1))

    if cost_fn is not None:
        return [del_n, costs]
    
    return del_n

# set up an online STCR reconstruction
def online_STCR_ISTA_timed(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, time_recon=200, cost_fn=None):

    start_time = time.time()
    running = True

    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)

    niter = 0

    while running:
        zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        inner_iter = 1 

        for i_i in range(inner_iter):
            deln = soft_threshold(deln - step_size *(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu)), lambdat*step_size)
            #cost = norm(E * deln - yn + E*xn_1) + norm((cp.sqrt(mu) * (G * deln)) - (cp.sqrt(mu) * (zn-etan-(G*xn_1)))) + (lambdat * (cp.sum(cp.abs(deln))))
        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
        
        cp.cuda.stream.get_current_stream().synchronize()
        elapsed = time.time() - start_time

        niter = niter+1
        if elapsed*1000 > time_recon:
            running = False
        

    print(f"niter {niter}")
    if cost_fn is not None:
        return [deln, costs]
    return deln


# set up an online STCR reconstruction
def online_STCR_ISTA(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, niter=20, cost_fn=None):
    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)
    for n_i in range(niter):
        zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        inner_iter = 1 
        """
        # cg instead?
        Af = lambda x: (E.H * E * x + (mu * (G.H * G * x)))
        y = (E.H * (yn + E*xn_1)) + (mu * (G.H * (zn - etan - (G*xn_1))))
        deln = cg(Af, y, deln, niter=inner_iter)
        """
        for i_i in range(inner_iter):
            deln = soft_threshold(deln - step_size *(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu)), lambdat*step_size)
            #cost = norm(E * deln - yn + E*xn_1) + norm((cp.sqrt(mu) * (G * deln)) - (cp.sqrt(mu) * (zn-etan-(G*xn_1)))) + (lambdat * (cp.sum(cp.abs(deln))))
        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
    if cost_fn is not None:
        return [deln, costs]
    return deln


# set up an online STCR reconstruction
def online_STCR_ISTA_2(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, niter=20, cost_fn=None):

    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)
    pogm_dict_outer = None
    for n_i in range(niter):
        gradout = lambda x: (zn - etan - (G*(deln+xn_1)))
        step_size_outer = 1

        inner_iter = 2 
        pogm_dict_inner = None
        for i_i in range(inner_iter):
            gradf = lambda x:(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu) )
            [deln, pogm_dict_inner] = FISTA_iteration(i_i, deln, gradf, step_size, pogm_dict_inner, lambdat*step_size, inner_iter)

        #zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        [zn, pogm_dict_outer] = FISTA_iteration(n_i, zn, gradout, step_size_outer, pogm_dict_outer, lambdas/mu, niter)

        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
    if cost_fn is not None:
        return [deln, costs]
    return deln

# set up an online STCR reconstruction
def online_STCR_ISTA_2_timed(E, G, xn_1, Ahyn, lambdat, lambdas, step_size, mu=1 , yn=None, deln=None, time_recon=200, cost_fn=None):

    start_time = time.time()
    running = True
    n_i = 0

    costs = []
    if deln is None:
       deln = cp.zeros(xn_1.shape, Ahyn.dtype)
    # initialize etan
    zn = G*(xn_1 + deln)
    etan = cp.zeros(zn.shape, Ahyn.dtype)
    pogm_dict_outer = None
    while running:
        gradout = lambda x: (zn - etan - (G*(deln+xn_1)))
        step_size_outer = 1

        inner_iter = 1 
        pogm_dict_inner = None
        for i_i in range(inner_iter):
            gradf = lambda x:(E.H * E * (deln+xn_1) - Ahyn + (G.H*(G*(deln+xn_1) - zn + etan)*mu) )
            [deln, pogm_dict_inner] = FISTA_iteration(i_i, deln, gradf, step_size, pogm_dict_inner, lambdat*step_size, inner_iter)

        #zn = soft_threshold(G*(deln+xn_1) + etan, lambdas/mu)
        [zn, pogm_dict_outer] = FISTA_iteration(n_i, zn, gradout, step_size_outer, pogm_dict_outer, lambdas/mu, 100)

        etan = etan - zn + (G*(deln+xn_1))
        if cost_fn is not None:
            costs.append(cost_fn(deln))
        
        cp.cuda.stream.get_current_stream().synchronize()
        elapsed = time.time() - start_time
        n_i = n_i + 1
        if elapsed*1000 > time_recon:
            running = False
    logging.debug(f"niter {n_i}, time elapsed {elapsed*1000} ms")
    if cost_fn is not None:
        return [deln, costs]
    return deln



def crop_half_FOV(image, dims=(0,1), size=None):
    h,w = (image.shape[dims[0]], image.shape[dims[1]])
    if size is None:
        crop_h = h // 2
        crop_w = w // 2
        size = [crop_h, crop_w]

    start_indices = [
        (h - size[0]) // 2,
        (w - size[1]) // 2
    ]

    print(start_indices)
    print(size)

    # Create a slice object for each dimension
    slices = [slice(None) for i in range(len(image.shape))]
    slices[dims[0]] = slice(start_indices[0], start_indices[0] + size[0])
    slices[dims[1]] = slice(start_indices[1], start_indices[1] + size[1])

    # slices = [slice(start_indices[0], start_indices[0] + size[0]),
    #           slice(start_indices[1], start_indices[1] + size[1])]

    # Use numpy's advanced indexing to crop the image
    return image[tuple(slices)]


def soft_threshold(x, threshold):
    x_phase = cp.angle(x)
    # do I need cp.sign(x)? Probably only if it is a real numpber. maybe, 'x_phase' covers both.
    return cp.maximum(cp.abs(x) - threshold, 0)  * cp.exp(1j*x_phase)


def remove_zero_padding(arr, axis):
    """
    Remove zero padding along a specified axis in a NumPy array.

    Parameters:
    arr (np.ndarray): Input array with potential zero padding.
    axis (int): Axis along which to remove zero padding.

    Returns:
    np.ndarray: Array with zero padding removed along the specified axis.
    """
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"Invalid axis {axis} for array with {arr.ndim} dimensions.")
    
    # Create a boolean mask where True represents non-zero slices
    non_zero_mask = np.any(arr != 0, axis=tuple(i for i in range(arr.ndim) if i != axis))
    
    # Use the boolean mask to slice the array
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None)  # Keep the full range for the axis we are slicing
    arr_sliced = arr[tuple(slices)]
    
    # Now use the boolean mask to filter out zero-only slices
    arr_trimmed = arr_sliced.take(np.where(non_zero_mask)[0], axis=axis)
    
    return arr_trimmed

def analyticaldcf(trajectory, adc_dwell=1e-6, ns=1):
    kxx = np.array(trajectory[:,0])
    kyy = np.array(trajectory[:,1])
    kzz = np.array(trajectory[:,2])

    # kx = kx[ndiscard:,0]
    # ky = ky[ndiscard:,0]
    gx = np.diff(np.concatenate(([0], kxx)), axis=0)/adc_dwell/42.58e6
    gy = np.diff(np.concatenate(([0], kyy)), axis=0)/adc_dwell/42.58e6

    # Analytical DCF formula
    # 1. Hoge RD, Kwan RKS, Bruce Pike G. Density compensation functions for spiral MRI. 
    # Magnetic Resonance in Medicine. 1997;38(1):117-128. doi:10.1002/mrm.1910380117
    cosgk = np.cos(np.arctan2(kxx, kyy) - np.arctan2(gx, gy))
    w = np.sqrt(kxx*kxx+kyy*kyy)*np.sqrt(gx*gx+gy*gy)*np.abs(cosgk)
    w[-int(ns//2):] = w[-int(ns//2)] # need this to correct weird jump at the end and improve SNR
    w = w/np.max(w)
    return w

def smooth(img, box=5, use_cpu=False):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''
    if use_cpu:
        t_real = np.zeros(img.shape)
        t_imag = np.zeros(img.shape)

        ndimage.filters.uniform_filter(img.real,size=box,output=t_real)
        ndimage.filters.uniform_filter(img.imag,size=box,output=t_imag)
    else:
        img_cp = cp.array(img)
        t_real = cp.zeros(img.shape)
        t_imag = cp.zeros(img.shape)
        cpsp.uniform_filter(img_cp.real,size=box,output=t_real)
        cpsp.uniform_filter(img_cp.imag,size=box,output=t_imag)

    simg = t_real + 1j*t_imag
    
    return simg
    

def calculate_csm_walsh_gpu(img, smoothing=5, niter=3):
    '''Calculates the coil sensitivities for 2D data using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the eigenvector power method (default ``3``)

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]``
    '''
    odim = img.ndim
    if img.ndim == 3:
        img = img[:,np.newaxis,:,:] # add a z dim
    #assert img.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"
    img = cp.array(img)
    ncoils = img.shape[0]
    nz = img.shape[1]
    ny = img.shape[2]
    nx = img.shape[3]
    
    start_time = time.time()

    # Compute the sample covariance pointwise
    Rs = cp.zeros((ncoils,ncoils,nz,ny,nx),dtype=img.dtype)
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:,:] = img[p,:,:,:] * cp.conj(img[q,:,:,:])
            
    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q] = smooth(Rs[p,q,:,:,:], smoothing)
    
    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    rho = cp.zeros((nz, ny, nx))
    csm = cp.zeros((ncoils, nz, ny, nx),dtype=img.dtype)
    v = cp.sum(Rs,axis=0)
    lam = cp.linalg.norm(v,axis=0)
    vo = v/lam
    #for z in range(nz):
    if(odim==3):
        vo = vo.squeeze()
        Rs = Rs.squeeze()
        for x in range(nx):
            v = vo[:,:,x]
            
            for iter in range(niter):
                v = cp.transpose(cp.matmul(cp.transpose(Rs[:,:,:,x],[2,0,1]),v),[1,0,2])
                lam = cp.linalg.norm(v,axis=0)
                v = v/lam
                v = cp.diagonal(v,axis1=1,axis2=2)
            
            rho[:,:,x] = cp.diagonal(lam)
            csm[:,0,:,x] = v

    else:
        for y in range(ny):
            for x in range(nx):
                v = vo[:,:,y,x]
                for iter in range(niter):
                    v = cp.transpose(cp.matmul(cp.transpose(Rs[:,:,:,y,x],[2,0,1]),v),[1,0,2])
                    lam = cp.linalg.norm(v,axis=0)
                    v = v/lam
                    v = cp.diagonal(v,axis1=1,axis2=2)
                
                rho[:,y,x] = cp.diagonal(lam)
                csm[:,:,y,x] = v
            
    del v,lam,Rs
    
    return (cp.asnumpy(csm), cp.asnumpy(rho))