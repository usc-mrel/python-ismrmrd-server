from timeit import default_timer as timer
import numpy as np
import cupy as cp
import math

# Helper functions
def dmd(X, Y, r):
	"""
	Compute the Dynamic Mode Decomposition of the data matrix X and Y with rank r.
	"""
	# Compute the SVD of X
	U, S, V = cp.linalg.svd(X, full_matrices=False) 	# Economy SVD. Returns U, S, Vh

	# Compute the truncated SVD of X
	Ur = U[:, :r]
	Sr_inv = cp.linalg.pinv(cp.diag(S[:r])) #cp.diag(1 / S[:r])
	Vr = V[:r, :]
	Vr = Vr.T.conj()

	# Compute the A matrix
	A = Ur.T.conj() @ Y @ Vr @ Sr_inv
	# Compute the (1) eigenvalues and (2) eigenvectors of A
	# Cannot do this in GPU, go back to CPU	
	A = cp.asnumpy(A)
	L, Q = np.linalg.eig(A)

	L = cp.array(L)
	Q = cp.array(Q)

	# Compute the DMD modes
	Phi = Y @ Vr @ Sr_inv @ Q

	# Compute the DMD amplitudes
	b = cp.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]
	
	return Phi, L, b



def dmd_filt(Igrid, dt, f_threshold):
	
	"""
	Filter the data I using the DMD method.
	"""

	Nx = Igrid.shape[0]
	Nf = Igrid.shape[2]
	
	I = Igrid.reshape(Nx*Nx, Nf)
	# Actual DMD
	X = I[:, :-1]
	Y = I[:, 1:]
	Phi, L, b = dmd(X, Y, Nf-1)
	f = cp.arctan2(L.imag, L.real) / (2 * math.pi * dt)

	# Time Modulations
	t 	= cp.arange(Nf)
	Ltb = b.reshape((-1,1)) * (L.reshape((-1,1)) ** t.reshape((1,-1)))

	# Filtering
	f_idx = cp.hstack( (cp.where(cp.abs(f) > f_threshold)[0], cp.where(abs(L) < 0.5)[0]) )
	S = Phi[:, f_idx] @ Ltb[f_idx, :]
	R = I - Phi @ Ltb

	I_filt_ = I - S
	return I_filt_.reshape((Nx, Nx, Nf))