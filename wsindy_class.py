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

# Rescaling currently requires
# - all poly library
# - special derivatives

# Special derivatives (e.g. grad) in apply_operator
# dont yet give proper Laplace-Beltrami ops in other coords


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
  init_guess_x,t = for critical k: [x0, y0, m1, m2]
  max_its = specify maximum number of MSTLS iterations
  noise = noise ratio of artifical Gaussian noise
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
    # Set the symbolic derivative library
    d_0 = [d for d in self.lhs.keys()]
    d_rhs = [d for d in self.library.keys()]
    self.derivatives = d_0 + d_rhs

    # Set the multi-index matrix
    self.set_alpha()

    # Set the number of derivatives in the library
    self.S = len(self.derivatives)

    # Set the largest derivatives along each axis
    self.alpha_bar = tuple(map(max, zip(*self.alpha)))
    return


  def set_alpha(self):
    """
    Sets multi-indices for needed derivatives.
    """
    # Derivative "matrix" of partial derivative indices
    self.alpha = []
    self.special_derivs = False
    for string in self.derivatives:

      # No derivatives, (0,...,0)
      if string == '1':
        self.alpha.append((self.D+1) * (0,))

      # Individual partials ∂x_i = ẟ_ij
      elif string.startswith('dd'):
        xyz = ''.join(self.axis_names)
        d_i = (self.D+1)*[0]
        for char in string[2:]:
          i = xyz.find(char)
          d_i[i] += 1
        self.alpha.append(tuple(d_i))

      # Gradient or divergence, ∇ = (1,...,1,0)
      elif (string == 'grad') or (string == 'div'):
        self.special_derivs = True
        for d in range(self.D):
          self.alpha.append(d*(0,)+(1,)+(self.D-d)*(0,))

      # Laplacian, Δ = (2,...,2,0)
      elif string == 'lap':
        self.special_derivs = True
        for d in range(self.D):
          self.alpha.append(d*(0,)+(2,)+(self.D-d)*(0,))

    self.alpha = tuple((self.alpha[0],)) + tuple(set(self.alpha[1:]))
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
    rescale_condition = self.poly_library and not self.special_derivs
    defaults.setdefault('rescale', rescale_condition)
    defaults.setdefault('scales', None)
    defaults.setdefault('verbosity', True)
    defaults.setdefault('init_guess_x', [15, 1, 10, 0])
    defaults.setdefault('init_guess_t', [2, 1, 15, 0])
    defaults.setdefault('max_its', None)
    defaults.setdefault('noise', 0)
    defaults.setdefault('sparsify', ['original','scaled'][rescale_condition])
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

    # If requested, add artifial noise
    if self.noise != 0:
      self.add_noise()

    # Get Jacobian determinant for chosen coord system
    self.set_jacobian()
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
    An n-choose-k function that accepts non-integer inputs.
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
      for f_j in self.library[d_i]:
        for field in f_j['fields']:

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


  def add_noise(self):
    """
    Adds artificial i.i.d. Gaussian noise to observations.
    """
    # Use field magnitudes to set variance of Gaussian noise
    U_rms = [(torch.sqrt((u**2).mean())).item() for u in self.fields]
    for i, u_i in enumerate(self.fields):
      sigma = self.noise * U_rms[i]
      noise = torch.normal(mean=0, std=sigma, size=u_i.shape)
      u_i += noise


  def set_jacobian(self):
    """
    Sets the Jacobian determinant for the chosen coordinate system.
    Currently accepts:
      - 'Cartesian'
      - '2D_spherical'
    """
    # Get coordinate system
    coords = self.coord_system

    # Accepted values
    if coords not in ['Cartesian', '2D_spherical']:
      raise ValueError(f'set_jacobian: {coords} coordinates not supported.')
    elif coords == 'Cartesian':
      self.jacobian = 1
    elif coords == '2D_spherical':
      _,Y,_ = torch.meshgrid(self.X[0], self.X[1], self.X[2], indexing='ij')
      cos = torch.cos(Y)
      #a = (1e3)*6371 # Mean Earth radius
      #self.jacobian = (a**2) * cos
      self.jacobian = cos
    return


  def wsindy(self):
    """
    The main Weak SINDy model discovery algorithm.
    For a given 'wsindy' class object, this method does the following:
      - computes an appropriate set of test functions (ψ),
      - builds the response vector (b) and weak library (G),
      - solves the resulting linear system (b=Gw) using MSTLS,
      - prints the results as desired.
    """
    # Test function creation
    self.test_fcns = self.get_test_fcns()

    # Build linear system (b = Gw)
    self.b = self.create_b()
    self.G = self.create_G()

    # MSTLS optimization
    self.weights, self.thresh_star, self.loss_star = self.MSTLS()

    # Print the results
    self.print_wsindy_results()
    return


  def get_test_fcns(self):
    """
    Returns test functions and any required derivatives.
    Currently uses separable Berstein polynomial test functions.
    """
    # Compute each component of separable test functions
    test_fcns = []
    for d in range(self.D + 1):

      # If requested, rescale grid spacings
      if self.rescale:
        space_scales = self.scales[1:self.D][0] + [self.scales[-1]]
        dx = space_scales[d] * self.dX[d]
      else:
        dx = self.dX[d]

      # Initialize discretized grids
      test_fcns_d = torch.zeros(len(self.alpha), 2*self.m[d]+1)
      unit_grid = torch.arange(-1, 1.0001, 1/self.m[d])

      # Precompute symbolic derivatives
      x_sym = sp.Symbol('x')
      phi_bar = (1 - x_sym**2)**self.p[d]
      vec = np.vectorize(self.D_phibar)

      # Compute axis-d components of the test functions
      alpha_d = tuple(deriv[d] for deriv in self.alpha)
      for i in range(len(self.alpha)):

        # Check for repeated values
        if (i > 0) and (alpha_d[i-1] == alpha_d[i]):
          test_fcns_d[i,:] += test_fcns_d[i-1,:]
        else:
          eval = torch.from_numpy(vec(unit_grid, alpha_d[i], x_sym, phi_bar))

        # Transform from unit grid to actual grid
        test_fcns_d[i,:] += (1/((self.m[d]*dx)**alpha_d[i])) * eval

      # Add axis-d components to the list
      test_fcns.append(test_fcns_d)

    # Mesh-grid plot
    if self.verbosity:
      self.plot_test_fcns(test_fcns)
    return test_fcns


  def D_phibar(self, x, D, x_sym, phi_bar):
    """
    Symbolically computes and evaluates test function derivatives.
    """
    # D-th derivative of degree p test function
    D_phi = sp.diff(phi_bar, x_sym, D)

    # Evaluate at point x
    if abs(x) < 1:
      return float(D_phi.subs(x_sym, x))
    else:
      return 0.0


  def plot_test_fcns(self, test_fcns):
    """
    Plots the first few (<= 3) test functions over the (x,y) axes.
    """
    num_plots = max(len(test_fcns)-1, 3)
    fig, axs = plt.subplots(1, num_plots, figsize=(6, 2))
    for i in range(num_plots):
      psi = [test_fcns[n][i+1, :] for n in range(len(test_fcns))]
      psi_x, psi_y = torch.meshgrid(psi[0], psi[1], indexing='ij')
      axs[i].imshow(psi_x * psi_y, cmap='coolwarm', interpolation='gaussian')
      axs[i].set_title('$\mathcal{D}^'+f'{i+1} \psi$')
      axs[i].axis('off')
    plt.tight_layout()
    self.plots.append(fig)
    plt.close(fig)
    return


  def create_b(self):
    """
    Returns the weak form response vector (b).
    """
    # Get operator and field
    operator = list(self.lhs.keys())[0]
    field_name = self.lhs[operator]
    field = getattr(self, field_name)

    # Apply evolution operator
    b = self.apply_operator(operator, field, field_name)

    # Subsample over query points and return
    b = (b[self.query_points]).reshape(-1,1)
    return b


  def create_G(self):
    """
    Returns the weak form library (G).
    """
    # Initialize library
    G = []

    # Loop over each operator and function in the library
    for operator in tqdm(self.library.keys()):
      for f_j in self.library[operator]:
        for field_names in f_j['fields']:
          field_name = [name for name in field_names]
          #field = [getattr(self,name) for name in field_name]
          field = []
          G_i = self.apply_operator(operator, field, field_name, fcn=f_j)
          G.append((G_i[self.query_points]).reshape(-1,1))

    # Concatenate into a single matrix
    G = torch.cat(G, dim=1)
    return G


  def apply_operator(self, operator, field, field_name, **kwargs):
    """
    Applies 'operator' to 'field' through test function convolutions.
    Currently supports the following differential operators:
      - the identity ('1')
      - partial derivatives ∂x_i ('ddx', 'ddy', 'ddxxy', etc.)
      - gradient ∇ ('grad')
      - laplacian Δ ('lap')
    -----------------------------------------------------------------------
    KWARGS:
    fcn = apply a library function to 'field' before computing convolutions
    -----------------------------------------------------------------------
    """
    import scipy.signal as sig

    # Rescale the grid and fields
    if self.rescale:
      yu, yx, yt = self.scales
      dX = [yx[d]*self.dX[d] for d in range(self.D)] + [yt*self.dX[-1]]

      # Apply function (if provided)
      if 'fcn' in kwargs:
        fcn = kwargs['fcn']
        handle = [key for key in fcn.keys()][0]
        arg_list = [tuple(yu[name]*getattr(self,name) for name in field_name)]
        U = list(map(fcn[handle], *zip(*arg_list)))[0]
      else:
        U = yu[field_name] * field.clone()

    else:
      # Normal grid spacing
      dX = self.dX

      # Apply function w/o rescaling
      if 'fcn' in kwargs:
        fcn = kwargs['fcn']
        handle = [key for key in fcn.keys()][0]
        arg_list = [tuple(getattr(self,name) for name in field_name)]
        U = list(map(fcn[handle], *zip(*arg_list)))[0]
      else:
        U = field.clone()

    # Match requested operator with test function derivatives
    # (Operator should already exist in 'self.derivatives')
    indices = self.get_operator_indices(operator)
    tf_list = []
    for index in indices:
      tf_list.append([tf[index,:].numpy() for tf in self.test_fcns])

    # Peform convolutions, looping over each partial derivative
    result = 0*U
    for test_fcns in tf_list:
      conv = U.numpy()

      for d in range(self.D+1):
        # Reshape test function appropriately
        slicing = [None] * len(test_fcns)
        slicing[d] = slice(None)

        # 1D convolution along the d-th axis
        conv = sig.convolve(conv, test_fcns[d][tuple(slicing)], mode='same')

      # Sum the different actions of the operators
      conv = self.jacobian * np.prod(dX) * torch.from_numpy(conv)
      result += conv

    return result


  def get_operator_indices(self, operator):
    """
    Find position of differential operators (e.g., 'ddx') in 'self.alpha'.
    Does the reverse of the 'set_alpha()' method.
    """
    indices = []
    if operator == '1':
       d_i = (self.D+1)*(0,)
       indices.append(self.alpha.index(tuple(d_i)))

    elif operator.startswith('dd'):
      xyz = ''.join(self.axis_names)
      d_i = (self.D+1)*[0]
      for char in operator[2:]:
        i = xyz.find(char)
        d_i[i] += 1
      indices.append(self.alpha.index(tuple(d_i)))

    elif operator == 'grad':
      for index, item in enumerate(self.alpha):
        if sum(item[:self.D])==1 and all(d_i==0 for d_i in item[self.D:]):
          indices.append(index)

    elif operator == 'lap':
      for index, item in enumerate(self.alpha):
        if sum(item[:self.D]).count(2)==1 and all(d_i==0 for d_i in item[self.D:]):
          indices.append(index)

    return indices


  def MSTLS(self):
    """
    Modified sequential thresholding least squares optimization routine.
    """
    import torch.linalg as la

    # Which field are we discovering a PDE for?
    key = [key for key in self.lhs.keys()][0]
    field_name = self.lhs[key]
    field = getattr(self, field_name)

    # Maximum iterations
    if self.max_its is None:
      self.max_its = self.G.shape[1]

    # Unique scaling factor
    if self.rescale:
      yu_n = self.scales[0][field_name]
      self.mu = (1/yu_n) * self.mu

    # Initial LS solution
    self.w_ls = la.lstsq(self.G, self.b, driver='gelsd').solution

    # Did user provide a threshold value?
    if self.threshold is not None:
      lambda_star = self.threshold

    # If not, loop over search space of candidate thresholds
    else:
      loss_history = []
      for lambda_n in self.lambdas:
        w_n, loss_n = self.MSTLS_iterate(lambda_n)
        loss_history.append(loss_n)

    # Find the loss-minimizing thresholding
    ind_star = loss_history.index(min(loss_history))
    lambda_star = self.lambdas[ind_star].item()

    # Use the best threshold to find the sparse weights
    w_star, loss_star = self.MSTLS_iterate(lambda_star)
    return w_star, lambda_star, loss_star


  def MSTLS_iterate(self, threshold):
    """
    Inner MSTLS loop, for a given threshold parameter.
    """
    import torch.linalg as la

    # Check sparsification data
    if self.sparsify == 'original' and self.rescale:
      mu = self.mu.unsqueeze(1)
      w_ls = mu * self.w_ls
    else:
      w_ls = self.w_ls.clone()

    # Compute |b|/|Li| bounds for all columns
    norm_b = la.norm(self.b)
    norm_Gi = la.norm(self.G, dim=0)
    bounds = norm_b / norm_Gi

    # Rescale the bounds if necessary
    if self.sparsify == 'original' and self.rescale:
      bounds = bounds * self.mu

    # Define upper and lower bounds
    L_bounds = threshold * torch.maximum(bounds,torch.ones(bounds.shape[0]))
    U_bounds = (1/threshold) * torch.minimum(bounds,torch.ones(bounds.shape[0]))

    # Begin applying iterative thresholding on elements of weight vector
    iteration = 0
    w_n = w_ls.clone()
    inds_old = torch.tensor([0])
    while iteration <= self.max_its:

      # Find in-bound and out-of-bound indices and set them to zero
      ib_inds = torch.where((abs(w_n[:,0])>=L_bounds)&(abs(w_n[:,0])<=U_bounds))[0]
      oob_inds = torch.where((abs(w_n[:,0])<L_bounds)|(abs(w_n[:,0])>U_bounds))[0]

      # Check stopping condition
      if (torch.equal(inds_old, ib_inds) and iteration!=0) or (ib_inds.shape[0]==0):
        break

      # Find LS solution amongst sparser, in-bound indices
      w_n[ib_inds] = la.lstsq(self.G[:,ib_inds], self.b, driver='gelsd').solution

      # Mask oob columns of G
      w_n[oob_inds] = 0

      # Unscale sparse solution if needed
      if self.sparsify == 'original' and self.rescale:
        w_n = mu * w_n

      inds_old = ib_inds
      iteration += 1
      if iteration == self.max_its:
        print('MSTLS reached the maximum number of iterations allowed.')

    # MSTLS loss is computed on scaled data, but returns unscaled weights
    if self.sparsify == 'original' and self.rescale:
      loss_n = self.MSTLS_loss((1/mu)*w_n, (1/mu)*w_ls)

    elif self.sparsify == 'original' and not self.rescale:
      loss_n = self.MSTLS_loss(w_n, w_ls)

    elif self.sparsify == 'scaled':
      loss_n = self.MSTLS_loss(w_n, w_ls)
      w_n = self.mu.unsqueeze(1) * w_n

    return w_n, loss_n


  def MSTLS_loss(self, weights, w_ls):
    """
    Returns MSTLS loss for a candidate threshold.
    """
    import torch.linalg as la

    # Least squares term
    LS_num = la.norm(torch.matmul(self.G, weights-w_ls)).item()
    LS_denom = la.norm(torch.matmul(self.G, w_ls)).item()
    LS_term = LS_num / LS_denom

    # Zero norm term
    zero_norm = sum(weights != 0).item() / weights.shape[0]

    # Return total loss
    loss_n = LS_term + zero_norm
    return loss_n

  
  def print_wsindy_results(self):
    """
    Prints results of the Weak SINDy routine.
    """
    # Print the discovered PDE
    self.term_names = self.get_term_names()
    self.pde = self.get_model()
    if self.verbosity:
      print(f'Discovered model: {self.pde}')

    # Print the explained variance
    self.compute_stats()
    return
  
  def get_term_names(self):
    """
    Returns the symbolic names of terms in the library (G) as strings.
    """
    # Evolution operator
    d_0 = [key for key in self.lhs.keys()][0]
    lhs_names = d_0 + '(' + self.lhs[d_0] + ')'

    # Library terms
    rhs_names = []
    for d_i in self.library.keys():
      for f_j in self.library[d_i]:
        for field in f_j['fields']:
          string = d_i + '[' + f_j['name'] + str(field) + ']'
          rhs_names.append(string)
    
    term_names = (lhs_names, rhs_names)
    return term_names
  
  def get_model(self):
    """
    Returns discovered PDE model symbolically, as a string.
    """
    # Find library terms with nonzero weights
    lhs_names, rhs_names = self.term_names
    inds = torch.nonzero(self.weights[:,0])[:,0].tolist()

    # Append discovered terms onto model
    model = ['({0:.2f})*'.format(self.weights[i,0].item())+rhs_names[i] for i in inds]
    model = lhs_names + ' = ' + ' + '.join(model)
    return model
  

  def compute_stats(self):
    """
    Sets the residuals and explained variance of the discovered WSINDy model.
    """
    # Rescale weights if necessary
    if (self.sparsify=='original' and self.rescale) or (self.sparsify=='scaled'):
      w = (1/self.mu.unsqueeze(1)) * self.weights
    elif self.sparsify == 'original' and not self.rescale:
      w = self.weights

    # Compute vector of residuals
    b = self.b.numpy()
    G = self.G.numpy()
    w = w.numpy()
    self.residuals = b - (G @ w)

    # Explained variance
    uv = np.sum(self.residuals[:,0]**2) / np.sum((b[:,0]-np.mean(b[:,0]))**2)
    ev = 100*(1 - uv)
    self.explained_variance = ev
    if self.verbosity:
      print(f'Sparse model explains {round(ev,3)}% of the data\'s variance.')

    # ADD AUTOMATIC HISTOGRAM OF RESIDUALS?
    return
