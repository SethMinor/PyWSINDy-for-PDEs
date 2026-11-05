# HELPER FUNCTIONS

import torch
import scipy
import numpy as np
import itertools
import symengine as sp

import torch.linalg as la
from scipy.signal import convolve
from scipy.special import factorial
import matplotlib.pyplot as plt
from tqdm import tqdm

def changepoint(x, x0, y0, m1, m2):
  """
  Defines a two-piece linear function:
  - changepoint at (x0,y0)
  - slope = m1 for x < x0
  - slope = m2 for x >= x0
  """
  return np.piecewise(x, [x<x0], [lambda x:m1*(x-x0)+y0, lambda x:m2*(x-x0)+y0])

def F_root(m,k,N,tau_hat,tau):
  """
  Root-finding helper function used for spectral matching.
  [See Appendix A of Messenger, Bortz (JCP, 2021) for details.]
  """
  log_term = np.log((2*m-1)/m**2)
  mid_term = (2*np.pi*k*m)**2 - 3*(tau_hat*N)**2
  last_term = 2*(tau_hat*N)**2 * np.log(tau)
  return log_term * mid_term - last_term

def subsample(s, m, x):
  """
  Returns indices which subsample a 1D tensor "x" with step size "s" and padding "m".
  """
  if (2*m + 1) > x.shape[0]:
    raise ValueError('Error: m produces non-compact support.')
  return list(range(m, x.shape[0]-m, s))

def reshape_subsampled_tensor(Uk, mask, s, m, X):
  """
  Returns a subsampled tensor "Uk" to its native dimensions.
  """
  xk = [len(subsample(s[i], m[i], X[i])) for i in range(len(X))]
  return Uk.reshape(tuple(xk))

def my_nchoosek(n, k):
  """
  Computes n-choose-k for potentially non-integer k.
  """
  n_factorial = scipy.special.factorial(n)
  k_factorial = scipy.special.factorial(np.ceil(k))
  nk_term = scipy.special.factorial(n-np.floor(k))
  return n_factorial / (nk_term * k_factorial)

def sep_convolve(u, kernels, volume_element=1.):
  """
  FFT-based convolution with separable kernel.
  """
  conv = volume_element * u.clone()
  for i, Ki in enumerate(kernels):
    shape = u.ndim * [1]
    shape[i] = -1
    conv = convolve(conv, Ki.reshape(shape), mode='valid')
  return conv

# Returns residuals and R^2
def compute_residuals(G, w, b):
  r = b - G@w
  R2 = 1 - (r**2).sum() / ((b - b.mean())**2).sum()
  return r,R2

# Add noise
def add_noise(U, sigma_NR):
  if sigma_NR == 0:
    return U
  else:
    U_rms = (torch.sqrt((U**2).mean())).item();
    sigma = sigma_NR * U_rms
    epsilon = torch.normal(mean=0, std=sigma, size=U.shape, dtype=torch.float64)
    return U + epsilon