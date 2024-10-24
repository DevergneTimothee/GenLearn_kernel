from sklearn.gaussian_process.kernels import RBF
from kooplearn.data import traj_to_contexts
from kooplearn._src.linalg import _rank_reveal, modified_QR, weighted_norm
from kooplearn._src.utils import fuzzy_parse_complex, topk
import numpy as np
import scipy
import scipy.sparse
from scipy.integrate import romb
from scipy.special import binom
from scipy.stats.sampling import NumericalInversePolynomial

def return_M(kernel,X, friction,k_X):
    """
    Function that returns the M matrix as in the paper
    """
    sigma = kernel.length_scale
        
    X = X.reshape(X.shape[0],-1)

    difference = (X[:, np.newaxis, :] - X[np.newaxis, :, :])

    n =  difference.shape[2]
    dphi = (1/sigma**2 -difference[:,:,0]**2  / sigma**4  )* k_X     
    return friction * dphi

def return_N(kernel, X, friction, k_X):
    """
    Function that returns the M matrix as in the paper
    """
    sigma = kernel.length_scale    
    X = X.reshape(X.shape[0],-1)
    difference = (X[:, np.newaxis, :] - X[np.newaxis, :, :])
    n =  difference.shape[2]
    dphi =  difference[:,:,0] * k_X / sigma**2     

    return np.sqrt(friction) * dphi


def compute_eigvecs(eta, k_X, N,u,v):
    """ 
    Return eigenvectors of the resolvent
    """
    return np.sqrt(eta)*k_X @ u + N @v


def rrr_UV17(k_X, N, M, eta, gamma, r):
    """
    Computes the U and V matrices of the rrr estimator of the resolvent
    """
    L = np.block([[k_X/eta, np.zeros(k_X.shape)],[np.zeros(k_X.shape),np.zeros(k_X.shape)]])
    n= k_X.shape[0]
    k_gamma = k_X + gamma * np.eye(n)
    J= k_X - N@np.linalg.inv(M+gamma*eta*np.eye(n))@N.T

    sigmas_2, vectors = scipy.sparse.linalg.eigs(J@k_X, k=r+5,M=(J+gamma*np.eye(n))*eta)
    indices = np.argsort(sigmas_2.real)[-r:]

    V_global = vectors[:,indices]
    V_global =V_global@ np.diag(1/np.sqrt(np.diag(V_global.T @ k_X @V_global)))
    sigma = np.diag(sigmas_2[indices])


    make_U = np.block([np.eye(n)/np.sqrt(eta),-N @ np.linalg.inv( M + eta * gamma * np.eye( n ) ) ] ).T
    U_global= make_U @(k_X@V_global -eta*V_global@sigma)/ (gamma*eta)

    evals, eigvecs_l, eigvecs_r = scipy.linalg.eig(V_global.T@V_global@sigma,left=True)

    lambdas = eta - 1/evals
    #indices = np.argsort(lambdas)
    eigvecs_l /= np.sqrt(evals)
    return lambdas, U_global, V_global, eigvecs_l, eigvecs_r, np.sqrt(sigmas_2[indices])


def compute_s(eta, k_X, N, M,U, v,sigmas,ev_rrr):
    """
    Computes the spectral bias of the estimator
    """
    indice = np.argsort(ev_rrr)[-2]
    print(indice)
    print(ev_rrr[indice])
    F = np.block([[eta*k_X, np.sqrt(eta)*N],[np.sqrt(eta)*N.T,M]])
    test = v[:,indice].T@U.T@F@U@v[:,indice]
    return test * sigmas[indice]

def make_fit(sigma_kernel, gamma_rrr, eta, train_data, rank,friction):
    """
    Computes the rrr estimator of the resolvent
    """
    signs = [1,1,-1]
    print(eta)
    kernel = RBF(length_scale=sigma_kernel)
    n = train_data.shape[0]

    k_X = kernel(train_data)/n
    M = return_M(kernel, train_data, friction, k_X)
    N = return_N(kernel, train_data, friction, k_X)
    ev_rrr, U, V, u,v, sigmas = rrr_UV17(k_X, N, M, eta, gamma_rrr, rank)
    eigvecs = compute_eigvecs(eta,k_X,N, (U@v)[:n,:],(U@v)[n:])
    indices = np.argsort(ev_rrr)
    sorted_ev = ev_rrr[indices]
    sorted_eigvecs =eigvecs[:,indices]

    return sorted_ev, sorted_eigvecs, compute_s(eta,k_X, N, M,U, v, sigmas,ev_rrr)

