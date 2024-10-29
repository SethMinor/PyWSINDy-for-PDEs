# WEAK SINDY MODULE

import torch
import torch.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import re
#%pip install symengine
import symengine as sp
from tqdm.notebook import tqdm

# CURRENT ISSUES:
# yu currently all set to '1'
# not plotting query points


class wsindy:
  """
  USER INPUTS
  -----------
  Data = [(name, tensor), ... ]
    - name = tensor name (string)
    - tensor = scalar field data

  LHS, Library = [(name, fcn, [∂'s]), ... ]
    - name = function name (string)
    - fcn = candidate library functions (lambda)
    - [∂'s] = derivatives to take (list of strings)

  Domain = [(name, bounds), ... ]
    - name = variable name (string)
    - bounds = [a,b], with a and b floats (list)


  HYPERPARAMETERS
  ---------------
  Grid and variable definition
  ----------------------------
  # x,t = spatial, temporal domains
  # dx,dt = spatial, temporal discretizations
  coord_system = 'Cartesian', etc.

  Weak SINDy parameters
  ---------------------
  m = explicit (mx,...,mt) values
  s = explicit (sx,...,st) values
  lambdas = MSTLS threshold search space
  threshold = known optimal threshold
  p = explicit (px,...,pt) values
  tau = test function tolerance
  tau_hat = Fourier test function tolerance
  rescale = scale the data? (True or False)
  scales = explicit ([yu],[yx],yt) scaling factors
  M = explicit scaling matrix

  Miscellaneous
  -------------
  verbosity = report info and create plots? (0 or 1)
  init_guess = for critical wavenumber (x0, y0, m1, m2)
  max_its = specify maximum number of MSTLS iterations
  sigma_NR = noise ratio of artifical gaussian noise
  sparsify = use 'original' or 'scaled' data in MSTLS
  field_names = variable names, e.g., ['u','v','w']
  augment = append operator? E.g., ('Div','2d_cartesian')
  """

  # Constructor
  def __init__(self, data, lhs, library, domain, **kwargs):

    # Scalar fields and their names
    field_names, fields = [], []
    for i in range(len(data)):
      name, field = data[i]
      field_names.append(name)
      fields.append(field)
      setattr(self, name, field)        # sets attributes (u,...,v)
    self.field_names = field_names      # strings ['u',...,'v']
    self.fields = fields                # tensors [u,...,v]
    self.lhs = lhs
    self.library = library
    self.domain = domain

    # Grid-like parameters
    self.D = fields[0].dim() - 1        # spatial dimension
    self.shape = tuple(fields[0].shape) # sets attributes (Nx,...,Nt)
    x_names, X, dX = [], [], []
    for i,axis in enumerate(self.domain):
      name, bounds = axis[0], axis[1]
      x_names.append(name)
      xn = torch.linspace(bounds[0], bounds[1], self.shape[i])
      X.append(xn)
      delta = (xn[1] - xn[0]).item()
      dX.append(delta)
      setattr(self, name, xn)           # sets attributes (x,...,t)
      setattr(self, 'd'+name, delta)    # sets attributes (dx,...,dt)
    self.x_names = x_names              # strings ['x',...,'t']
    self.X = X                          # tensors [x,...,t]
    self.dX = dX                        # floats [dx,...,dt]

    # Derivative and function libraries
    self.ddt, self.derivatives, self.alpha_bar = self.get_derivatives()
    self.beta_bar = self.get_max_powers()
    self.libsize = sum(len(item[-1]) for item in self.library)

    # Default parameter settings
    defaults = {'coord_system': 'Cartesian',
                's': [max(int(s/25),1) for s in self.shape],
                'lambdas': 10**((4/49)*torch.arange(0,50)-4),
                'threshold': None,
                'tau': 1E-10,
                'tau_hat': 1,
                'rescale': True,
                'verbosity': 0,
                'init_guess_x': [15, 1, 10, 0],
                'init_guess_t': [2, 1, 15, 0],
                'max_its': None,
                'sparsify': 'scaled',
                'm': 'default',
                'p': 'default'}

    # Set user-defined parameters, otherwise use defaults
    for key, default in defaults.items():
      setattr(self, key, kwargs.get(key, default))

    # Automatic computation of test function support (m), degree (p)
    # (Uses first field by default.)
    if 'm' not in kwargs:
      m = []
      for d in range(self.D):
        m.append(self.get_support(fields[0], d))    # Space
      m.append(self.get_support(fields[0], self.D)) # Time
      self.m = m
    if 'p' not in kwargs:
      p = []
      for d in range(self.D + 1):
        alpha_bar_d = self.alpha_bar[d]
        log_tau_term = np.ceil(np.log(self.tau)/np.log((2*self.m[d]-1)/self.m[d]**2))
        p.append(max(log_tau_term, alpha_bar_d + 1))
      self.p = p
    
    # Rescale for better conditioning
    if self.rescale == True:
      # Compute scaling factors
      if 'scales' not in kwargs:
        self.scales = self.get_scales()
      # Get scaling matrix
      self.mu = self.get_scale_matrix()
    
    # Define query points
    self.query_points = self.get_query_points()

  
  """
  DERIVATIVE LIBRARY
  """
  def get_derivatives(self):
    # Unique derivatives
    ddt, derivatives = set(), set()
    for item in self.lhs:
      ddt.update(item[-1])         # Evolution operator
    for item in self.library:
      derivatives.update(item[-1]) # Library derivatives
    # Turn into "matrix" of tuples
    ddt = {string: self.count_derivative(string) for string in ddt}
    derivatives = {string: self.count_derivative(string) for string in derivatives}
    # Find maximum(s)
    alpha_bar = list(ddt.values()) + list(derivatives.values())
    alpha_bar = tuple(max(position) for position in zip(*alpha_bar))
    return ddt, derivatives, alpha_bar
    
  def count_derivative(self, string):
    # No derivative
    if string == '1':
      return (self.D+1)*(0,)
    # Gradient, divergence
    elif (string == 'grad') or (string == 'div'):
      return self.D*(1,) + (0,)
    # Laplacian
    elif string == 'lap':
      return self.D*(2,) + (0,)
    else: 
      # Start counting letters after the 'd' in the string
      counts = dict()
      for name in self.x_names:
        counts[name] = 0
      if string.startswith('d'):
        for char in string[1:]:
          if char in counts:
            counts[char] += 1
      # Return counts in x,...,t order
      multi_index = tuple()
      for name in self.x_names:
        multi_index += (counts[name],)
      return multi_index
  

  """
  FUNCTION LIBRARY
  """
  def get_max_powers(self):
    # Count highest monomial power for each variable
    max_powers = dict()
    for term, _, _ in self.library:
      matches = re.findall(r"([a-zA-Z])(?:\^(\d+))?", term)
      for var, power in matches:
        # Default power is 1 (if not recognized)
        power = int(power) if power else 1
        if var in max_powers:
          max_powers[var] = max(max_powers[var], power)
        else:
          max_powers[var] = power
    return max_powers
  

  """
  COMPUTE TEST FUNCTION SUPPORT
  """
  def changepoint(self, x, x0, y0, m1, m2):
    return np.piecewise(x, [x<x0], [lambda x:m1*(x-x0)+y0, lambda x:m2*(x-x0)+y0])
    
  def support_loss(self, m, k, N):
    log_term = np.log((2*m-1)/m**2)
    mid_term = (2*np.pi*k*m)**2 - 3*(self.tau_hat*N)**2
    last_term = 2*(self.tau_hat*N)**2 * np.log(self.tau)
    return log_term * mid_term - last_term
  
  def get_support(self, U, d):
    # Get critical wavenumber (k)
    # (Separates signal from noise dominated modes.)
    Uhat_d = abs(torch.fft.rfft(U, n=self.shape[d], dim=d))
    Uhat_d = Uhat_d.mean(dim=[dim for dim in range(Uhat_d.ndimension()) if dim!=d])
    Hd = torch.cumsum(Uhat_d, dim=0)
    Hd = (Hd / Hd.max()).numpy()
    freqs = torch.arange(0, self.shape[d]//2 + 1, 1).numpy()
    if d != self.D:
      p0 = self.init_guess_x
    else:
      p0 = self.init_guess_t
    params = scipy.optimize.curve_fit(self.changepoint, freqs, Hd, p0=p0)[0]
    k = int(params[0])

    # Find 'm' as a root of 'support_loss'
    p0 = (np.sqrt(3) * self.shape[d] * self.tau_hat) / (2*np.pi*k)
    p0 = 0.5*p0*(1 + np.sqrt(1 - (8/np.sqrt(3)) * np.log(self.tau)))
    md = int(scipy.optimize.root(self.support_loss, p0, args=(k, self.shape[d])).x[0])
    if self.verbosity == 1:
      plt.figure(figsize=(4, 2))
      plt.plot(freqs, Uhat_d, 'r--',
               label='$\mathcal{F}^'+f'{self.x_names[d]}[{self.field_names[0]}]$')
      plt.axvline(x = k)
      plt.xlabel(f'Wavenumber, $k$ ($k_c = {k}$)')
      plt.title(f'Support $m_{self.x_names[d]} = {md}$')
      plt.legend(loc = 'best')
      plt.show()
    return md


  """
  RESCALING THE LIBRARY COLUMNS
  """
  def my_nchoosek(self, n, k):
    n_factorial = scipy.special.factorial(n)
    k_factorial = scipy.special.factorial(np.ceil(k))
    nk_term = scipy.special.factorial(n-np.floor(k))
    return n_factorial / (nk_term * k_factorial)

  def get_scales(self):
    # Compute L2 norm of U and U^beta
    beta_bar = list(self.beta_bar.values())
    N = len(beta_bar)
    U_2 = [la.norm(u.reshape(-1)).item() for u in self.fields]
    U_b = [la.norm((self.fields[n]**beta_bar[n]).reshape(-1)).item() for n in range(N)]

    # Compute scales using ansatz given in pape
    #yu = [(U_2[n]/U_b[n])**(1/beta_bar[n]) for n in range(N)]
    yu = [1 for n in range(N)]
    yx = [(1/(self.m[d]*self.dX[d])) * (self.my_nchoosek(self.p[d],self.alpha_bar[d]/2)
          *scipy.special.factorial(self.alpha_bar[d]))**(1/self.alpha_bar[d]) for d in range(self.D)]
    yt = (1/(self.m[-1]*self.dX[-1])) * (self.my_nchoosek(self.p[-1], self.alpha_bar[self.D]/2)
          *scipy.special.factorial(self.alpha_bar[self.D]))**(1/self.alpha_bar[self.D])
    return yu, yx, yt
  
  def get_scale_matrix(self):
    yu,yx,yt = self.scales
    alpha = tuple(a.ddt.values()) + tuple(a.derivatives.values())
    mu = []
    for i in range(len(alpha)):
      # Exponents for [yx], yt
      yx_exps = [yx[d]**(alpha[0][d] - alpha[i][d]) for d in range(self.D)]
      yx_term = np.prod(yx_exps)
      t_exp = alpha[0][-1] - alpha[i][-1]

      # Product of field scaling factors
      #yu_term = [yu[n]**(fj['poly'][j][n]) for n in range(len(U))]
      #yu_term = np.prod(yu_term)
      yu_term = 1

      # Set corresponding mu value
      mu.append(yu_term * yx_term * (yt**t_exp))
    mu = torch.tensor(mu)
    return mu

  """
  DEFINE QUERY POINTS
  """
  def uniform_subsample(self, s, m, x):
    if (2*m + 1) > x.shape[0]:
      raise Exception('Error: m produces non-compact support.')
    xk = x[m:-m:s]
    indices = (x.unsqueeze(0) == xk.unsqueeze(1)).nonzero(as_tuple=True)[1]
    return indices.tolist()

  def get_query_points(self):
    D_ = self.D + 1
    subsamples = [self.uniform_subsample(self.s[i%D_], self.m[i%D_], self.X[i%D_])
                  for i in range(D_)]
    QP_mask = list(itertools.product(*subsamples))
    query_points = tuple(zip(*QP_mask))
    return query_points
