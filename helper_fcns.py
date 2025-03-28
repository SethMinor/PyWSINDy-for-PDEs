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

# Two-piece linear approximation
def changepoint(x, x0, y0, m1, m2):
    return np.piecewise(x, [x<x0], [lambda x:m1*(x-x0)+y0, lambda x:m2*(x-x0)+y0])

# Define the scalar-valued function F(m)
def F_root(m,k,N,tau_hat,tau):
  log_term = np.log((2*m-1)/m**2)
  mid_term = (2*np.pi*k*m)**2 - 3*(tau_hat*N)**2
  last_term = 2*(tau_hat*N)**2 * np.log(tau)
  return log_term * mid_term - last_term

# Compute test function degrees given support radii
def compute_degrees(d, m, alpha, tau=1.e-10):
  alpha_bar = max(tuple(item[d] for item in alpha))
  log_tau_term = np.ceil(np.log(tau)/np.log((2*m-1)/m**2))
  p = int(max(log_tau_term, alpha_bar + 1))
  return p

# Indices of query points along a single axis
def subsample(s, m, x):
  if (2*m + 1) > x.shape[0]:
    raise ValueError('Error: m produces non-compact support.')
  return list(range(m, x.shape[0]-m, s))

# Return subsampled tensor to normal dimensions
def reshape_subsampled_tensor(Uk, mask, s, m, X):
  xk = [len(subsample(s[i], m[i], X[i])) for i in range(len(X))]
  return Uk.reshape(tuple(xk))

# Carefully compute n-choose-k with non-integer k
def my_nchoosek(n, k):
  n_factorial = scipy.special.factorial(n)
  k_factorial = scipy.special.factorial(np.ceil(k))
  nk_term = scipy.special.factorial(n-np.floor(k))
  return n_factorial / (nk_term * k_factorial)

# Calculate symbolic derivatives of test fcns (Dth derivative at degree p)
def D_phibar(x, D, x_sym, phi_bar):
  D_phi = sp.diff(phi_bar, x_sym, D)
  if abs(x) < 1.:
    return float(D_phi.subs(x_sym, x))
  else:
    return 0.

# FFT-base convolution with separable kernel
def sep_convolve(u, kernels):
  conv = u.clone()
  for i, Ki in enumerate(kernels):
    shape = u.ndim * [1]
    shape[i] = -1
    conv = convolve(conv, Ki.reshape(shape), mode='valid')
  return conv

# Compute a weak polynomial term, <D_phi, u^power>
def compute_weak_poly(u, kernels, spacing, power=1., yu=1., yxyt=1.):
  weak_poly = torch.from_numpy(sep_convolve((yu*u)**power, kernels))
  weak_poly *= yxyt * np.prod(spacing)
  return weak_poly

# Weak multivariable polynomial term, <D_phi, u1^p1 * ... * un^pn>
def compute_weak_multipoly(u, kernels, spacing, power=[1.], yu=[1.], yxyt=1.):
  assert type(u) == type(power) == type(yu) == list, "Must provide a list."
  monomial = 1
  for i,ui in enumerate(u):
    monomial *= (yu[i]*ui)**power[i]
  weak_poly = torch.from_numpy(sep_convolve(monomial, kernels))
  weak_poly *= yxyt * np.prod(spacing)
  return weak_poly

# Compute a weak trig term, <D_phi, cos(au+b)>
def compute_weak_trig(u, kernels, spacing, freq=1., phase=0., yxyt=1.):
  trig = torch.cos(freq*u + phase)
  weak_trig = torch.from_numpy(sep_convolve(trig, kernels))
  weak_trig *= yxyt * np.prod(spacing)
  return weak_trig

# Computes loss for a given candidate threshold
def loss(w_n, w_LS, G):
  LS_num = la.norm(G @ (w_n - w_LS)).item()
  LS_denom = la.norm(G @ w_LS).item()
  LS_term = LS_num / LS_denom
  zero_norm = sum(w_n != 0).item()/w_n.shape[0]
  loss_n = LS_term + zero_norm
  return loss_n

# Prints the symbolic PDE
def symbolic_pde(lhs_name, rhs_names, w):
  nonzero_inds = w.nonzero().flatten()
  nonzero_coeffs = w[nonzero_inds].tolist()
  nonzero_terms = [rhs_names[i] for i in nonzero_inds]

  pde = []
  for coeff, term in zip(nonzero_coeffs, nonzero_terms):
    if coeff >= 0.:
      pde.append(f"+ {coeff:.2f}{term}")
    else:
      pde.append(f"- {abs(coeff):.2f}{term}")
  pde = lhs_name + " = " + " ".join(pde)
  return pde

# Returns residuals and R^2
def compute_residuals(G, w, b):
  r = b - G@w
  R2 = 1 - (r**2).sum() / ((b - b.mean())**2).sum()
  return r,R2

# Add noise
def add_noise(U, sigma_NR):
  U_rms = (torch.sqrt((U**2).mean())).item();
  sigma = sigma_NR * U_rms
  epsilon = torch.normal(mean=0, std=sigma, size=U.shape, dtype=torch.float64)
  return U + epsilon
