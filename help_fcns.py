# HELPER FUNCTIONS

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
def reshape_subsampled_tensor(U, mask, s, m, X):
  xk = [len(subsample(s[i], m[i], X[i])) for i in range(len(X))]
  UK = U[mask].clone()
  return UK.reshape(tuple(xk))

# Compute scale for a state variable, yu
def compute_u_scale(u, beta_max):
  U_2 = la.norm(u).item()
  U_beta = la.norm(u**beta_max).item()
  yu = (U_2 / U_beta)**(1 / beta_max)
  return yu

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

# Compute the weak time derivative, <phi_t, u>
def compute_weak_dudt(u, d_phi_dt, spacing, yu=1., yxyt=1.):
  weak_dudt = torch.from_numpy(convolve(yu*u, d_phi_dt, mode='same'))
  weak_dudt *= yxyt * np.prod(spacing)
  return weak_dudt

# Compute the weak polynomial term, <D_phi, u^beta>
def compute_weak_poly(u, beta, D_phi, spacing, yu=1., yxyt=1.):
  weak_poly = torch.from_numpy(convolve((yu*u)**beta, D_phi, mode='same'))
  weak_poly *= yxyt * np.prod(spacing)
  return weak_poly

# Compute the weak trig term, <D_phi, cos(au+b)>
def compute_weak_trig(u, D_phi, spacing, freq=1., phase=0., yxyt=1.):
  trig = torch.cos(freq*u + phase)
  weak_trig = torch.from_numpy(convolve(trig, D_phi, mode='same'))
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
      pde.append(f"+ {coeff:.2f}*{term}")
    else:
      pde.append(f"- {abs(coeff):.2f}*{term}")
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
