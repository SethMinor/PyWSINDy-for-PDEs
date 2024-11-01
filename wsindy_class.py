# WEAK SINDY MODULE

import torch
import torch.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
#import re
#%pip install symengine
import symengine as sp
from tqdm.notebook import tqdm


class wsindy:
  """
  USER INPUTS
  -----------
  data = [u1, ..., un]
    - list of torch tensors

  lhs = {'operator': 'ui'}
    - read as "apply evolution operator to ui"
    - "operator" is an informative string (e.g., 'ddt')
    - "ui" is a string specifying a field (e.g., 'u1')

  library = {'operator_1': [fi,...,fj], ..., 'operator_S': [fn,...,fm]}
    - read as "apply operator_1 to the functions [fi,...,fj]"
    - each fj = {'function_type': lambda fcn, 'fields': list, 'power': tuple}
      - "function_type" is an informative string (e.g., 'poly')
        - the options are "poly", "poly_coeffs", or "gen_fcn"
        - for "poly", the user should also set "fj['power'] = (bi,...,bj)"

  HYPERPARAMETERS
  ---------------
  Grid and variable definition
  ----------------------------
  x,...,t = spatial, temporal domains
  dx,...,dt = spatial, temporal discretizations
  coord_system = 'Cartesian', '2D_spherical', etc.

  Weak SINDy parameters
  ---------------------
  m = test function support radii
  s = inter-query point distance
  lambdas = MSTLS threshold search space
  threshold = choose one MSTLS threshold
  p = test function degrees
  tau = test function compact support tolerance
  tau_hat = Fourier smoothing parameter
  rescale = rescale the data? (True or False)
  scales = scale invariance factors ([yu],[yx],yt)

  Miscellaneous
  -------------
  verbosity = print info and plots? (True or False)
  init_guess_x,t = for critical k, [x0, y0, m1, m2]
  max_its = specify maximum number of MSTLS iterations
  noise = noise ratio of artifical gaussian noise
  sparsify = use 'original' or 'scaled' data in MSTLS
  field_names = variable names, e.g., ['u','v','w']
  """

  # Constructor
  def __init__(self, data, lhs, library, **kwargs):

    # Set basic attributes
    self.fields = data
    self.lhs = lhs
    self.library = library

    # Set the names of the fields
    self.set_field_names(**kwargs)

    # Set the spatial and temporal grids
    self.set_domain(**kwargs)

    # Set the function and derivative choices
    self.set_derivatives()
    self.set_functions()

    # Set user-defined parameters, otherwise use defaults
    self.set_wsindy_params(**kwargs)


  def set_field_names(self, **kwargs):
    """
    Sets names of scalar fields.
    -------------------------------
    KWARGS:
    field_names = ['u1', ..., 'un']
    -------------------------------
    """
    # Number of scalar fields
    n = len(self.fields)

    # Check for user-defined names
    if 'field_names' in kwargs:
      self.field_names = kwargs['field_names'][0:n]
    else:
      self.field_names = ['u' + str(i+1) for i in range(n)]

    # Set each field as an attribute
    for i in range(n):
      setattr(self, self.field_names[i], self.fields[i])
    return


  def set_domain(self, **kwargs):
    """
    Sets attributes related to the discrete (x,...,t) grid.
    -------------------------------------------------------
    KWARGS:
    axis_names = ['x',...,'t']
    domain = [(0,Lx),...,(0,T)]
    -------------------------------------------------------
    """
    # Spatial dimension and tensor shape
    self.D = self.fields[0].dim() - 1
    self.shape = tuple(self.fields[0].shape)

    # Check for user-defined names
    if 'axis_names' in kwargs:
      self.axis_names = kwargs['axis_names'][0:self.D+1]
    else:
      self.axis_names = ['x','y','z'][0:self.D] + ['t']

    # Check for user-defined domain
    if 'domain' in kwargs:
      self.domain = kwargs['domain']
    else:
      self.domain = (self.D+1)*[(0,1)]

    # Build discretized grid
    self.X, self.dX = [], []
    for i in range(self.D+1):
      # Axis
      x_i = torch.linspace(self.domain[i][0], self.domain[i][1], self.shape[i])
      self.X.append(x_i)
      setattr(self, self.axis_names[i], x_i)

      # Grid spacing
      dx_i = (x_i[1] - x_i[0]).item()
      self.dX.append(dx_i)
      setattr(self, 'd'+self.axis_names[i], dx_i)
    return


  def set_derivatives(self):
    """
    Sets attributes related to the derivative library.
    """
    # Set the number of derivatives in the library
    self.S = len(self.library)

    # Set the symbolic derivative library
    d_0 = [d for d in self.lhs.keys()]
    d_rhs = [d for d in self.library.keys()]
    self.derivatives = d_0 + d_rhs

    # Set the multi-index matrix
    self.set_alpha()

    # Set the largest derivatives along each axis
    self.alpha_bar = tuple(map(max, zip(*self.alpha)))
    return 


  def set_alpha(self):
    """
    Sets multi-indices for needed derivatives.
    """
    # Derivative "matrix" of partial derivative indices
    self.alpha = []
    for string in self.derivatives:

      # No derivatives, (0,...,0)
      if string == '1':
        self.alpha.append((self.D+1) * (0,))

      # Gradient or divergence, ∇ = (1,...,1,0)
      elif (string == 'grad') or (string == 'div'): 
        self.alpha.append(self.D*(1,) + (0,))
      
      # Laplacian, Δ = (2,...,2,0)
      elif string == 'lap':
        self.alpha.append(self.D*(2,) + (0,))
      
      # Individual partials ∂x_i = ẟ_ij
      else:
        xyz = ''.join(self.axis_names)
        d_i = (self.D+1)*[0]
        if string.startswith('dd'):
          for char in string[2:]:
            i = xyz.find(char)
            d_i[i] += 1
        self.alpha.append(tuple(d_i))
    self.alpha = tuple(self.alpha)
    return


  def set_functions(self):
    """
    Sets attributes related to the library functions.
    """
    # Find unique function terms
    functions = [fcn for fcns in self.library.values() for fcn in fcns]
    functions = list({fcn['name']: fcn for fcn in functions}.values())
    self.functions = list(functions)
    self.J = sum(len(fcn['fields']) for fcn in self.functions)

    # Check if library includes only polynomials
    self.check_polynomial_library()

    # If it does, find the max powers
    if self.poly_library:
      self.beta_bar = {}
      for fcn in self.functions:
        fields, powers = fcn['fields'][0], fcn['power']
        for u_i, b_i in zip(fields, powers):
          if u_i in self.beta_bar:
            self.beta_bar[u_i] = max(self.beta_bar[u_i], b_i)
          else:
            self.beta_bar[u_i] = b_i
    return
  

  def check_polynomial_library(self):
    """
    Checks if the library functions are all monomials.
    """
    poly_library = all(list(fcn.keys())[0]=='poly' for fcn in self.functions)
    self.poly_library = poly_library
    return


  def set_wsindy_params(self, **kwargs):
    """
    Sets all user-defined and default wsindy hyperparameters.
    ---------------------------------------------------------
    KWARGS:
    Same kwargs as provided to the 'wsindy' class.
    ---------------------------------------------------------
    """
    # Define default wsindy hyperparameters
    defaults = {}
    defaults.setdefault('coord_system', 'Cartesian')
    defaults.setdefault('s', [max(int(s/25),1) for s in self.shape])
    defaults.setdefault('lambdas', 10**((4/49)*torch.arange(0,50)-4))
    defaults.setdefault('threshold', None)
    defaults.setdefault('tau', 1E-10)
    defaults.setdefault('tau_hat', 1)
    defaults.setdefault('rescale', self.poly_library)
    defaults.setdefault('scales', None)
    defaults.setdefault('verbosity', True)
    defaults.setdefault('init_guess_x', [15, 1, 10, 0])
    defaults.setdefault('init_guess_t', [2, 1, 15, 0])
    defaults.setdefault('max_its', None)
    defaults.setdefault('noise', 0)
    defaults.setdefault('sparsify', ['original','scaled'][self.poly_library])
    defaults.setdefault('m', [])
    defaults.setdefault('p', [])

    # Set user-defined parameters, otherwise use defaults
    for key, default in defaults.items():
      setattr(self, key, kwargs.get(key, default))
    
    # Keep plots for verbose output
    if self.verbosity:
      self.plots = []
    
    # Make sure test function support (m), degrees (p) are specified
    if self.m == []:
      for d in range(self.D):
        self.m.append(self.compute_tf_support(d))    # Space
      self.m.append(self.compute_tf_support(self.D)) # Time
    if self.p == []:
      for d in range(self.D+1):
        m_d = self.m[d]
        p_d = np.ceil(np.log(self.tau) / np.log((2*m_d - 1) / m_d**2))
        self.p.append(max(p_d, self.alpha_bar[d] + 1))
    
    # Set query points
    self.set_query_points()

    # If library is compatible, compute scaling factors
    if self.rescale and (self.scales == None):
      self.set_scales()
    
    # ADD ARTIFICIAL NOISE
    return
    
  
  def compute_tf_support(self, d):
    """
    Returns test function support radius on axis = d.
    The goal is to separate signal-dominanted modes from noise-dominated modes.
    """
    import torch.fft as fft
    import scipy.optimize as op

    # Set the field used for noise estimation
    U = self.fields[0]
    N = self.shape[d]

    # Averaged Fourier transform on axis = d
    fft_d = abs(fft.rfft(U, n = N, dim = d))
    avg_dims = [dim for dim in range(fft_d.dim()) if dim != d]
    fft_d = fft_d.mean(dim = avg_dims)

    # Normalized cumulative sum
    sum_d = torch.cumsum(fft_d, dim = 0)
    sum_d = (sum_d / sum_d.max()).numpy()

    # Estimate the critical wavenumber (k)
    if d != self.D:
      k0 = self.init_guess_x
    else:
      k0 = self.init_guess_t
    freqs = torch.arange(0, N//2 + 1, 1).numpy()
    params = op.curve_fit(self.changepoint, freqs, sum_d, p0=k0)[0]
    k = int(params[0])

    # Compute the support radius on axis = d
    m0 = (np.sqrt(3) * N * self.tau_hat) / (2 * np.pi * k)
    m0 = 0.5 * m0 * (1 + np.sqrt(1 - (8/np.sqrt(3)) * np.log(self.tau)))
    m_d = int(op.root(self.support_loss, m0, args = (k, N)).x[0])

    if self.verbosity:
      plt.figure(figsize=(4, 2))
      x_label = f'{self.axis_names[d]}[{self.field_names[0]}]$'
      plt.plot(freqs, fft_d, 'r--', label = '$\mathcal{F}^' + x_label)
      plt.axvline(x = k, label = f'$k_c = {k}$')
      plt.xlabel(f'Wavenumber, $k$')
      plt.title(f'Support $m_{self.axis_names[d]} = {m_d}$')
      plt.legend(loc = 'upper right')
      self.plots.append(plt.gcf())
      plt.close()
    return m_d


  def changepoint(self, x, x0, y0, m1, m2):
    """
    Piecewise linear function with a changepoint at (x0,y0).
    """
    return np.piecewise(x,[x<x0],[lambda x:m1*(x-x0)+y0,lambda x:m2*(x-x0)+y0])
  

  def support_loss(self, m, k, N):
    """
    Loss function for automatic computing test function support radii.
    """
    log_term = np.log((2*m - 1) / m**2)
    mid_term = (2*np.pi * k * m)**2 - 3*(self.tau_hat * N)**2
    last_term = 2 * (self.tau_hat * N)**2 * np.log(self.tau)
    return log_term * mid_term - last_term
  

  def set_query_points(self):
    """
    Sets a query point "mask" that subsamples the data.
    Formatted in a way that can be directly evaluated on library columns.
    """
    # Subsampling along each axis
    axes = self.D + 1
    sampled_inds = []
    for i in range(axes):
      s_i, m_i, x_i = self.s[i%axes], self.m[i%axes], self.X[i%axes]
      sampled_inds.append(self.subsample(s_i, m_i, x_i))
    
    # Create query point mesh
    query_points = list(itertools.product(*sampled_inds))
    self.query_points = tuple(zip(*query_points))
    return
  

  def subsample(self, s, m, x):
    """
    Uniformly subsamples indices along one axis.
    """
    if (2*m + 1) > x.shape[0]:
      raise ValueError('Error: chosen m produces non-compact support.')
    x_k = x[m:-m:s]
    indices = (x.unsqueeze(0) == x_k.unsqueeze(1)).nonzero(as_tuple=True)[1]
    return indices.tolist()

  
  def set_scales(self):
    """
    Sets scale invariance factors (γ's) and the change of variables matrix (μ).
    The goal is preserve physical scaling while improving condition numbers.
    """
    import scipy.special as sci

    # Fields with a nonzero power
    nz_fields = [getattr(self,key) for key in self.beta_bar.keys()]
    nz_powers = list(self.beta_bar.values())
    N = len(nz_fields)

    # Compute L2 norms of fields and their highest powers
    U_L2 = [la.norm(u).item() for u in nz_fields]
    Ub_L2 = [la.norm(nz_fields[n]**nz_powers[n]).item() for n in range(N)]

    # Compute scaling factors using ansatz given in the paper
    yu = [(U_L2[n]/Ub_L2[n])**(1/nz_powers[n]) for n in range(N)]
    yu = {key: yu[k] for k,key in enumerate(self.beta_bar.keys())}
    yx = [(1/(self.m[d] * self.dX[d])) * (self.my_nck(self.p[d], self.alpha_bar[d]/2)
          * sci.factorial(self.alpha_bar[d]))**(1/self.alpha_bar[d]) for d in range(self.D)]
    yt = (1/(self.m[-1] * self.dX[-1])) * (self.my_nck(self.p[-1], self.alpha_bar[self.D]/2)
          * sci.factorial(self.alpha_bar[self.D]))**(1/self.alpha_bar[self.D])
    self.scales = (yu, yx, yt)

    # Compute scaling matrix
    mu = self.get_scale_matrix()
    self.mu = mu
    return
  

  def my_nck(self, n, k):
    """
    An n-choose-k function that accepts non-integers.
    """
    import scipy.special as sci
    n_factorial = sci.factorial(n)
    k_factorial = sci.factorial(np.ceil(k))
    nk_term = sci.factorial(n - np.floor(k))
    return n_factorial / (nk_term * k_factorial)
  

  def get_scale_matrix(self):
    """
    Returns the diagonal change of variables matrix (μ).
    """
    # Scaling factors
    yu, yx, yt = self.scales

    # Build the matrix
    mu = []
    for i, d_i in enumerate(self.derivatives[1:], start=1):
      for j, f_j in enumerate(self.library[d_i]):
        for n, field in enumerate(f_j['fields']):

          # Derivative factors
          yx_exps = [(self.alpha[0][d] - self.alpha[i][d]) for d in range(self.D)]
          yx_term = np.prod([yx[d]**yx_exps[d] for d in range(self.D)])
          t_exp = self.alpha[0][-1] - self.alpha[i][-1]

          # Polynomial factors
          yu_term = [yu[uk]**f_j['power'][k] for k,uk in enumerate(field)]
          yu_term = np.prod(yu_term)

          # Set corresponding mu values
          mu.append(yu_term * yx_term * (yt**t_exp))

    mu = torch.tensor(mu)
    return mu

