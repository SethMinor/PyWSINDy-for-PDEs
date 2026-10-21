from helper_fcns_dev import *
import csv

# def symbolic_ode(lhs_name, rhs_names, w):
#   return symbolic_eqn(lhs_name, rhs_names, w)

class WSINDy:
  def __init__(self,
                U, # ............................................... State variable data (N_x x ... x N_t tensor)
                derivatives, # ....................... Derivative operators to include in library (multi-indices)
                monomials, # ..................... Candidate monomial terms to include in library (multi-indices)
                grid, # ............................. The dsicretized space-time grid (list of vectors [x,...,t])
                aux_vars = [], # ......................... Auxilliary variable(s) (list of tensors [V_1,V_2,...])
                equation_type = "pde", # .................................................. Either "ode" or "pde"
                volume_element = 1., # .................... Volume element of the space-time grid (dx * ... * dt)
                var_names = None, # .............. Names of state/aux variables (list of strings ['U','V_1',...])
                grid_axis_names = None, # .......... Names of spatiotemporal axes (list of strings ['x',...,'t'])
                rescale = True, # ...................... Controls whether scale-invariant preconditioning is used
                verbosity = True, # ...................... Controls whether additional plots and info are printed
                subsample_rate = None, # .............. Spacing between query points (list of ints [s_x,...,s_t])
                test_fcn_support_radii = None, # ............... Support grid widths (list of ints [m_x,...,m_t])
                test_fcn_poly_degree = None, # ........ 1/2 of test fcn poly degrees (list of ints [p_x,...,p_t])
                test_fcn_support_tol = 1e-10, # ......... Tolerance for test fcn magnitude at boundary of support
                test_fcn_spectrum_decay = 2, # ............. Spectral matching parameter (decay rate of spectrum)
                changepoint_guess = [10,1,10,0], # ...... For spectral matching changepoint [x,y,slope_1,slope_2]
              ):

    self.U = U
    self.derivatives = derivatives
    self.monomials = monomials
    self.aux_vars = aux_vars
    self.equation_type = equation_type
    self.volume_element = volume_element
    self.rescale = rescale
    self.verbosity = verbosity
    self.test_fcn_support_tol = test_fcn_support_tol
    self.test_fcn_spectrum_decay = test_fcn_spectrum_decay
    self.changepoint_guess = changepoint_guess

    # Error-handling for equation type
    if equation_type not in ('ode', 'pde'):
      raise ValueError("equation_type must be 'ode' or 'pde'.")
    if equation_type == 'ode':
      if self.U.dim() != 1:
        raise ValueError("ODE data must be one-dimensional in time.")
      if len(self.grid) != 1:
        raise ValueError("ODE data requires a single time axis.")
    elif self.U.dim() < 2:
      raise ValueError("PDE data requires at least one spatial axis.")

    # Strings for printing symbolic expressions
    self.var_names = var_names if var_names is not None else ['u']+['v'+str(i+1) for i in range(len(aux_vars))]
    self.grid_axis_names = grid_axis_names if grid_axis_names is not None \
      else [('x','y','z')[i] for i in range(U.ndim-1)] + ['t']
    self.derivative_names = self.get_derivative_names()

    self.grid = grid if type(grid) == list else [grid]
    self.spacing = [xi.diff()[0] for xi in self.grid]
    self.subsample_rate = subsample_rate if subsample_rate is not None else [U.shape[i]//50 for i in range(U.ndim)]

    # Test function parameters
    self.test_fcn_support_radii = test_fcn_support_radii if test_fcn_support_radii is not None \
      else self.compute_test_fcn_support_radii()
    self.test_fcn_poly_degree = test_fcn_poly_degree if test_fcn_poly_degree is not None \
      else self.compute_test_fcn_poly_degree()

    # Scale-invariant preconditioning
    if rescale:
      [self.x_scales, self.t_scale] = self.compute_spatial_scales()
      self.monomials_max = max([bj[0] for bj in monomials])
      self.u_scale = self.compute_u_scale(self.U, self.monomials_max)
      self.aux_scales = []
      for i,Vi in enumerate(aux_vars):
        monomials_max = max([bj[i+1] for bj in self.monomials])
        self.aux_scales.append(self.compute_u_scale(Vi, monomials_max))
    else:
      [self.x_scales, self.t_scale, self.u_scale, self.aux_scales] = 4*[None]

    # Test function construction
    [self.mask, self.conv_mask] = self.compute_query_points()
    [self.test_fcn_axes, self.test_fcn_kernels] = self.build_test_fcn_axes()
    self.test_fcns = self.build_test_fcns()

  #################################################################################################################
  # AUTOMATIC HYPERPARAMETER SELECTION AND TUNING
  #################################################################################################################

  def compute_test_fcn_support_radii(self, method='spectral_matching'):
    """
    Automatically selects support radii (bandwidths) for test functions.
    (Currently, only "spectral matching" is supported.)
    """
    if method == 'spectral_matching':
      return [self.spectral_matching(d, changepoint, F_root) for d in range(self.U.dim())]
    else:
      raise ValueError(f'Method "{method}" is not supported right now.')

  def compute_test_fcn_poly_degree(self):
    """
    Automatically selects test function degrees for given support radii.
    [See Equation (4.3) of Messenger, Bortz (JCP, 2021) for details.]
    """
    def compute_degrees(d, m, derivatives, test_fcn_support_tol=1.e-10):
      derivatives_bar = max(tuple(item[d] for item in derivatives))
      log_tau_term = np.ceil(np.log(test_fcn_support_tol)/np.log((2*m-1)/m**2))
      p = int(max(log_tau_term, derivatives_bar + 1))
      return p
    test_fcn_poly_degree = [compute_degrees(d, md, self.derivatives, test_fcn_support_tol=self.test_fcn_support_tol) \
      for d,md in enumerate(self.test_fcn_support_radii)]
    return test_fcn_poly_degree

  def spectral_matching(self, d, changepoint, F_root):
    """
    Implements the "spectral matching" method from Appendix A of Messenger, Bortz (JCP, 2021).
    """
    Nd = self.U.shape[d]
    Uhat_d = abs(torch.fft.rfft(self.U, n=Nd, dim=d))
    dims = [dim for dim in range(Uhat_d.ndimension()) if dim != d]
    Uhat_d = Uhat_d.mean(dim = dims) if dims else Uhat_d

    Hd = torch.cumsum(Uhat_d, dim=0)
    Hd = (Hd/Hd.max()).numpy() # Normalize for curve fitting

    # Solve change-point problem
    freqs = torch.arange(0, np.floor(Nd/2)+1, 1).numpy()
    params = scipy.optimize.curve_fit(changepoint, freqs, Hd, p0=self.changepoint_guess)[0]
    k = int(params[0])

    # Solve root-finding problem
    guess = (np.sqrt(3)*Nd*self.test_fcn_spectrum_decay)/(2*np.pi*k)
    guess = guess * (1 + np.sqrt(1 - (8/np.sqrt(3))*np.log(self.test_fcn_support_tol)))/2
    md = int(scipy.optimize.root(F_root, guess, \
      args=(k,Nd,self.test_fcn_spectrum_decay,self.test_fcn_support_tol)).x[0])

    # Optional plotting
    if self.verbosity:
      axis = self.grid_axis_names[d]
      plt.figure(figsize=(6,2.5))
      Uhat_d = Uhat_d.numpy()
      label = r'Spectrum, $|\mathcal{F}_t[u]|$' if self.equation_type == 'ode' \
        else fr'Spectrum, $|\mathcal{{F}}_{axis}[u]|$'
      plt.plot(freqs, Uhat_d, 'r--', label=label)
      plt.plot(freqs, Uhat_d.max()*Hd, 'k', label='Cumulative sum')
      plt.plot(freqs, Uhat_d.max()*changepoint(freqs,*params), 'g--')
      plt.plot(params[0], Uhat_d.max()*params[1], 'go', label='Changepoint, $k_d$')
      plt.xlabel('Wavenumber, $k$', fontsize=10)
      plt.title(f'Spectral matching (${axis}$-axis): $m_{axis}={md}$ and $k_{axis}={k}$', fontsize=12)
      plt.legend(loc = 'lower right', fontsize=10)
      plt.grid(True)
      plt.show()
    return md

  #################################################################################################################
  # SCALE-INVARIANT PRECONDITIONING
  #################################################################################################################

  def compute_u_scale(self, u, monomials_max):
    """
    Returns rescaling factor "u_scale" for the state variable "u".
    """
    U_2 = la.norm(u).item()
    U_monomials = la.norm(u**monomials_max).item()
    u_scale = (U_2 / U_monomials)**(1 / max(1,monomials_max))
    return u_scale

  def compute_spatial_scales(self):
    """
    Compute spatiotemporal scales "x_scales" and "t_scale".
    (Note: x_scales = [] for ODEs.)
    """
    D = len(self.spacing) - 1
    max_x = []
    for d in range(len(self.derivatives[0]) - 1):
      max_d = max(tuple(ai[d] for ai in self.derivatives))
      max_x.append(max(1,max_d))
    max_t = max(1,max(tuple(ai[-1] for ai in self.derivatives)))

    # Ansatz given in Equation (4.9) of Messenger, Bortz (JCP, 2021)
    x_scales = [(1/(self.test_fcn_support_radii[d]*self.spacing[d]) * (my_nchoosek(self.test_fcn_poly_degree[d], max_x[d]/2)
          * factorial(max_x[d]))**(1/max_x[d])).item() for d in range(D)]
    t_scale = (1 / (self.test_fcn_support_radii[-1] * self.spacing[-1]) * (my_nchoosek(self.test_fcn_poly_degree[-1], max_t/2)
          * factorial(max_t))**(1/max_t)).item()
    return x_scales,t_scale

  def compute_scale_matrix(self, powers, derivs):
    """
    Returns the diagonal of the rescaling matrix, M = diag(mu), defined such that w = M * w_rescaled.
    [Note: we use the inverse of Equation (4.8) from Messenger, Bortz (JCP, 2021), which has a typo/error.]
    """
    D = self.U.dim() - 1
    [x_scales, t_scale] = [self.x_scales, self.t_scale]
    u_scale = [self.u_scale] + self.aux_scales

    num_terms = len(powers)
    mu = torch.zeros(num_terms, dtype=torch.float64)
    for j in range(num_terms):
      aj = derivs[j]
      bj = powers[j]
      if (aj==None) or (bj==None):
        mu[j] = 1.
      else:
        x_scale_exps = [x_scales[d]**(self.derivatives[0][d] - self.derivatives[aj][d]) for d in range(D)]
        x_scale_term = np.prod(x_scale_exps)
        t_scale_term = t_scale**(self.derivatives[0][-1] - self.derivatives[aj][-1])

        u_scale_term = [u_scale[n]**(bj[n]) for n in range(len(u_scale))]
        u_scale_term = np.prod(u_scale_term)

        mu[j] = (u_scale_term * x_scale_term * t_scale_term)/self.u_scale
    return mu

  #################################################################################################################
  # TEST FUNCTION CONSTRUCTION
  #################################################################################################################

  def compute_query_points(self):
    """
    Returns query point indices in two forms:
    - "mask" is defined for Nx x ... x Nt tensors and ignores padding (i.e., U[mask][k] = U[x_k,...,t_k])
    - "conv_mask" is defined relative to convolution padding (i.e., G[conv_mask][k] = G[x_k-m_x,...,t_k-m_t])
    """
    subsamples = [subsample(self.subsample_rate[i], self.test_fcn_support_radii[i], self.grid[i]) \
      for i in range(self.U.ndim)]
    cartesian_prod = itertools.product(*subsamples)
    mask = tuple(map(torch.tensor, zip(*cartesian_prod)))
    conv_mask = tuple([mask[i]-self.test_fcn_support_radii[i] for i in range(self.U.ndim)])

    # Optional plotting
    if self.verbosity:
      if self.equation_type == 'ode':
        t = self.grid[0]
        tk = t[subsamples[0]]

        plt.figure(figsize=(6,2.5))
        plt.plot(t, self.U, 'r', label='$' + self.var_names[0] + '$')
        plt.scatter(tk, self.U[mask], color='black', marker='.', s=6, label='Query points')
        plt.xlabel('$t$')
        plt.title('$' + self.var_names[0] + '(t)$')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True)
        plt.show()
      else:
        [x1, x2] = [self.grid[0], self.grid[1]]
        [X1, X2] = np.meshgrid(self.grid[0], self.grid[1])
        [xk1, xk2] = [x1[subsamples[0]], x2[subsamples[1]]]
        [XK1, XK2] = [Xi.flatten() for Xi in np.meshgrid(xk1, xk2)]
        slice0 = (slice(None),slice(None)) + (self.U.dim()-2)*(0,)

        plt.figure(figsize=(5,5))
        plt.pcolormesh(X1, X2, self.U[slice0].T, cmap='coolwarm')
        plt.scatter(XK1, XK2, color='black', marker='.', s=2, label='Query points')
        plt.xlabel(fr'${self.grid_axis_names[0]}$')
        plt.ylabel(fr'${self.grid_axis_names[1]}$')
        title = fr'$({self.grid_axis_names[0]},{self.grid_axis_names[1]})$' if self.U.ndim==2 \
          else fr'$({self.grid_axis_names[0]},{self.grid_axis_names[1]},\dots)$'
        plt.title('$' + self.var_names[0] + '$' + title)
        plt.legend(loc='upper right', fontsize=10)
        plt.show()
    return mask, conv_mask

  # Compute separable component along d-th axis
  def get_weight_fcns(self, d):
    [x,test_fcn_support_radii,test_fcn_poly_degree] = [self.grid[d], self.test_fcn_support_radii[d], self.test_fcn_poly_degree[d]]
    dx = (x[1] - x[0]).item()

    if (test_fcn_support_radii > (len(x)-1)/2) or (test_fcn_support_radii <= 1):
      raise ValueError('Error: invalid test function support.')

    if self.rescale:
      scale_d = (self.x_scales + [self.t_scale])[d]
      dx *= scale_d

    # Initialize grid of discretized test fcn values
    test_fcns_d = torch.zeros(len(self.derivatives), 2*test_fcn_support_radii+1, dtype=torch.float64)
    n_grid = torch.linspace(-1, 1, 2*test_fcn_support_radii+1, dtype=torch.float64)
    multi_index_d = tuple(ai[d] for ai in self.derivatives)

    x_sym = sp.Symbol('x')
    phi_bar = (1 - x_sym**2)**test_fcn_poly_degree
    vec = np.vectorize(D_phibar)

    for i in range(len(self.derivatives)):
      if (i > 0) and (multi_index_d[i-1] == multi_index_d[i]):
        test_fcns_d[i,:] += test_fcns_d[i-1,:]
      else:
        num_derivs = multi_index_d[i]
        A_i = torch.from_numpy(vec(n_grid, num_derivs, x_sym, phi_bar))
        test_fcns_d[i,:] += (1/((test_fcn_support_radii*dx)**num_derivs)) * A_i

    if self.verbosity:
      plt.figure(figsize=(6,2.5))
      for i in range(len(test_fcns_d[:,0])):
        plt.plot(test_fcn_support_radii*dx*n_grid, test_fcns_d[i,:], '--.', label=f'$i={i}$')
      plt.title(f'Axis $d={d}$ Test Functions')
      if self.equation_type == 'ode':
        plt.xlabel('$t_k - t$')
        plt.ylabel(r'$\mathcal{D}^i\phi(t_k - t)$')
      else:
        plt.xlabel('$x_k - x$')
        plt.ylabel(r'$\mathcal{D}^i\phi_d(x_k - x)$')
      plt.grid(True)
      plt.legend(loc='upper right')
      plt.show()
    return test_fcns_d

  def build_test_fcn_axes(self):
    test_fcn_axes = [self.get_weight_fcns(d) for d in range(len(self.test_fcn_support_radii))]
    test_fcn_kernels = [[test_fcn_axes[d][i,:] for d in range(self.U.ndim)] for i in range(len(self.derivatives))]
    return test_fcn_axes, test_fcn_kernels

  # Performs Knrocker product
  def build_test_fcns(self):
    num_derivs = len(self.test_fcn_axes[0])
    D = len(self.test_fcn_axes)
    test_fcns = []
    for s in range(num_derivs):
      test_fcn_axes = [self.test_fcn_axes[d][s,:] for d in range(D)]
      ijk = [chr(105 + i) for i in range(D)]
      einsum = ','.join(ijk) + '->' + ''.join(ijk)
      D_phi = torch.einsum(einsum, *test_fcn_axes)
      test_fcns.append(D_phi)
    return test_fcns

  #################################################################################################################
  # CONSTRUCTION OF WSINDY LINEAR SYSTEM
  #################################################################################################################

  def build_lhs(self, lhs_name):
    kernel = self.test_fcn_kernels[0]
    if self.rescale:
      scales_product = np.prod(self.x_scales + [self.t_scale])
      lhs = compute_weak_poly(self.U, kernel, self.spacing, u_scale=self.u_scale, scales_product=scales_product, volume_element=self.volume_element)
    else:
      lhs = compute_weak_poly(self.U, kernel, self.spacing, volume_element=self.volume_element)
    b = lhs[self.conv_mask]
    self.lhs_name = lhs_name
    self.lhs = b
    return

  # Computes default monomial library terms
  def create_default_library(self):
    [G, powers, derivs, rhs_names] = [],[],[],[]
    state = [self.U] + self.aux_vars
    if self.rescale:
      u_scale = [self.u_scale] + self.aux_scales
      scales_product = np.prod(self.x_scales + [self.t_scale])
    else:
      u_scale = len(state) * [1.]
      scales_product = 1.

    for i,ai in enumerate(tqdm(self.derivatives[1:]), start=1):
      kernel = self.test_fcn_kernels[i]
      for j,bj in enumerate(self.monomials):
        if all(bjd==0 for bjd in bj) and any(aid!=0 for aid in ai):
          continue # Avoid derivatives of constant terms
        else:
          assert (len(bj)-1) == len(self.aux_vars), "Inconsistent number of powers!"
          derivs.append(i)
          powers.append(bj)
          term = compute_weak_multipoly(state, kernel, self.spacing, power=bj, u_scale=u_scale, \
            scales_product=scales_product, volume_element=self.volume_element)
          name = self.format_monomial(bj) + self.derivative_names[i]
          G.append(term[self.conv_mask])
          rhs_names.append(name)
    return G, powers, derivs, rhs_names

  def set_library(self, G, powers, derivs, rhs_names):
    G = torch.stack(G, dim=1)
    if G.shape[0] != len(self.mask[0]):
      raise ValueError("Library has inconsistent dimensions.")
    self.rhs_names = rhs_names
    self.library = G
    if self.rescale:
      self.mu = self.compute_scale_matrix(powers, derivs)
    return

  #################################################################################################################
  # SPARSE REGRESSION
  #################################################################################################################

  # Full MSTLS optimization routine, scans through Lambdas
  def MSTLS(self, Lambda=None, Lambdas=10**((4/49)*torch.arange(0,50)-4)):
    w_LS = la.lstsq(self.library, self.lhs, driver='gelsd').solution

    if Lambda is not None:
      Lambda_star = Lambda
    else:
      loss_history = []
      for Lambda_n in Lambdas:
        [_, loss_n] = self.MSTLS_iterate(Lambda_n.item(), w_LS.clone())
        loss_history.append(loss_n)

      # Find minimizer (smallest minimzer, if not unique)
      index = loss_history.index(min(loss_history))
      Lambda_star = Lambdas[index].item()

    [w_star, loss_star] = self.MSTLS_iterate(Lambda_star, w_LS.clone())
    if self.rescale:
      w_star = self.mu * w_star
    self.Lambda = Lambda_star
    self.loss = loss_star
    self.coeffs = w_star
    return w_star

  # Sequential thresholding least squares routine
  def MSTLS_iterate(self, Lambda_n, w_LS):
    G = self.library
    b = self.lhs
    max_its = G.shape[1]

    bounds = la.norm(b) / la.norm(G,dim=0)
    lower = Lambda_n * torch.maximum(bounds, torch.ones(bounds.shape[0]))
    upper = (1/Lambda_n) * torch.minimum(bounds, torch.ones(bounds.shape[0]))

    iteration = 0
    w_n = w_LS.clone()
    nonzero_inds = torch.tensor([])
    while iteration <= max_its:
      ib_inds = torch.where((abs(w_n) >= lower) & (abs(w_n) <= upper))[0]
      oob_inds = torch.where((abs(w_n) < lower) | (abs(w_n) > upper))[0]

      if (torch.equal(nonzero_inds, ib_inds)) or (ib_inds.shape[0]==0):
        break
      w_n[ib_inds] = la.lstsq(G[:,ib_inds], b, driver='gelsd').solution
      w_n[oob_inds] = 0
      nonzero_inds = ib_inds
      iteration += 1

    loss_n = loss(w_n, w_LS, G)
    return w_n, loss_n
  
  # Returns symbolic derivatives
  def get_derivative_names(self):
    spatial_dimension = self.U.dim() - 1
    derivative_names = []
    for elem in self.derivatives:
      if all(value == 0 for value in elem):
        derivative_names.append('')
      else:
        # (0+1)-D, (1+1)-D, (2+1)-D, (3+1)-D case-handling
        if spatial_dimension == 0:
          derivative_names.append('_{'+'t'*elem[0]+'}')
        elif spatial_dimension == 1:
          derivative_names.append('_{'+'t'*elem[1]+'x'*elem[0]+'}')
        elif spatial_dimension == 2:
          derivative_names.append('_{'+'t'*elem[2]+'x'*elem[0]+'y'*elem[1]+'}')
        elif spatial_dimension == 3:
          derivative_names.append('_{'+'t'*elem[3]+'x'*elem[0]+'y'*elem[1]+'z'*elem[2]+'}')
        else:
          raise ValueError("Whoah! Spatial dimension can only be: 1, 2, or 3.")
    return derivative_names

  # Fancy monomial formatting
  def format_monomial(self, bj):
    terms = []
    for d in range(len(bj)):
      if bj[d] == 0:
        continue
      if bj[d] == 1:
        terms.append(self.var_names[d])
      else:
        terms.append(f'{self.var_names[d]}^{bj[d]}')
    return '(' + ' '.join(terms) + ')' if terms else '(1)'

  # Prints a report of the WSINDy run
  def print_report(self):
    print('HYPER-PARAMETERS')
    print(f'test_fcn_support_radii = {self.test_fcn_support_radii}')
    print(f'test_fcn_poly_degree = {self.test_fcn_poly_degree}')
    print(f'subsample_rate = {self.subsample_rate}')
    if self.rescale:
      if self.equation_type == 'ode':
        print(f't_scale = {self.t_scale:.3f}')
      else:
        scales = self.x_scales + [self.t_scale]
        print('[x_scales, t_scale] = ' + str([float(f" {yi:.3f}") for yi in scales]))
      print(f'u_scale = {self.u_scale:.3f}')
      aux_scales = [float(f'{scale:.3f}') for scale in self.aux_scales]
      print(f'Aux. scales = {aux_scales}\n')
    else:
      print('Not rescaled.\n')

    print('LIBRARY')
    print(f'Num. query points = {self.library.shape[0]}')
    print(f'Num. terms = {self.library.shape[1]}')
    print(f'cond(G) = {la.cond(self.library):.3e}\n')

    print('RESULTS')
    eq_label = self.equation_type.upper()
    eqn = symbolic_eqn(self.lhs_name, self.rhs_names, self.coeffs)
    num_terms = self.coeffs.count_nonzero().item()
    if self.rescale:
      [r, R2] = compute_residuals(self.library, self.coeffs/self.mu, self.lhs)
    else:
      [r, R2] = compute_residuals(self.library, self.coeffs, self.lhs)
    print(f'{eq_label}: {eqn}')
    print(f'Nonzero terms = {num_terms}')
    print(f'Rel. L2 error = {la.norm(r)/la.norm(self.lhs):.3f}')
    print(f'R^2 = {R2:.3f}')
    print(f'Lambda = {self.Lambda:.3e}')
    print(f'Loss = {self.loss:.3f}')
    return

  # Sweep hyperparameters (test_fcn_support_radii, Lambda, rescale) and tabulate model results
  def hyperparameter_sweep(self, lhs_name=None, m_values=None, Lambdas=None,
                           rescales=(True, False), library_fcn=None, truth=None,
                           csv_path=None, plot=True):
    if lhs_name is None:
      lhs_name = getattr(self, 'lhs_name', self.var_names[0] + self.derivative_names[0])
    if m_values is None:
      m_values = []
      for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        mf = [int(round(factor*mi)) for mi in self.test_fcn_support_radii]
        mf = [min(max(mi,2), (self.U.shape[i]-1)//2) for i,mi in enumerate(mf)]
        if mf not in m_values:
          m_values.append(mf)
    else:
      m_values = [(self.U.ndim*[mi] if type(mi)==int else list(mi)) for mi in m_values]
    if Lambdas is None:
      Lambdas = 10**((4/49)*torch.arange(0,50)-4)
    Lambdas = [float(Li) for Li in Lambdas]

    keys = ['test_fcn_support_radii','test_fcn_poly_degree','Lambda','rescale','model','terms','coeffs','sparsity','loss','R2','rel_L2','cond_G']
    if truth is not None:
      keys.append('coeff_err')
    results = {key: [] for key in keys}

    subsample_rate = self.U.ndim * [1]
    for test_fcn_support_radii in tqdm(m_values):
      for rescale in rescales:
        model = WSINDy(self.U, self.derivatives, self.monomials, self.grid, aux_vars=self.aux_vars, var_names=self.var_names,
                       test_fcn_support_radii=test_fcn_support_radii, subsample_rate=subsample_rate, volume_element=self.volume_element, test_fcn_support_tol=self.test_fcn_support_tol, test_fcn_spectrum_decay=self.test_fcn_spectrum_decay,
                       changepoint_guess=self.changepoint_guess, verbosity=False, rescale=rescale,
                       equation_type=self.equation_type)
        if library_fcn is None:
          [G, powers, derivs, rhs_names] = model.create_default_library()
        else:
          [G, powers, derivs, rhs_names] = library_fcn(model)
        model.build_lhs(lhs_name)
        model.set_library(G, powers, derivs, rhs_names)
        cond_G = la.cond(model.library).item()

        for Lambda in Lambdas:
          w = model.MSTLS(Lambda=Lambda)
          w_tilde = w/model.mu if rescale else w
          [r, R2] = compute_residuals(model.library, w_tilde, model.lhs)
          nonzero = w.nonzero().flatten().tolist()

          results['test_fcn_support_radii'].append(str(test_fcn_support_radii))
          results['test_fcn_poly_degree'].append(str(model.test_fcn_poly_degree))
          results['Lambda'].append(Lambda)
          results['rescale'].append(rescale)
          results['model'].append(symbolic_eqn(lhs_name, rhs_names, w))
          results['terms'].append([rhs_names[j] for j in nonzero])
          results['coeffs'].append([w[j].item() for j in nonzero])
          results['sparsity'].append(len(nonzero))
          results['loss'].append(model.loss)
          results['R2'].append(R2.item())
          results['rel_L2'].append((la.norm(r)/la.norm(model.lhs)).item())
          results['cond_G'].append(cond_G)

          # Max relative coefficient error, E_inf in the paper
          if truth is not None:
            got = dict(zip(results['terms'][-1], results['coeffs'][-1]))
            err = max(abs(got.get(name, 0.) - cj)/abs(cj) for name,cj in truth.items())
            results['coeff_err'].append(err)

    if csv_path is not None:
      with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        for i in range(len(results['Lambda'])):
          writer.writerow([results[key][i] for key in keys])

    if plot:
      self.plot_sweep(results)
    return results

  # Heatmaps of sweep metrics over the (test_fcn_support_radii, Lambda) plane
  def plot_sweep(self, results, metrics=None):
    if metrics is None:
      metrics = ('sparsity', 'R2') + (('coeff_err',) if 'coeff_err' in results else ())
    m_labels = list(dict.fromkeys(results['test_fcn_support_radii']))
    Lambdas = list(dict.fromkeys(results['Lambda']))
    rescales = list(dict.fromkeys(results['rescale']))
    titles = {'sparsity': 'Nonzero terms', 'R2': '$R^2$', 'coeff_err': r'$\log_{10}$ Coeff. Error'}

    for rescale in rescales:
      fig, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 0.5*len(m_labels)+1.8))
      ax = np.atleast_1d(ax)
      for j,metric in enumerate(metrics):
        Z = np.full((len(m_labels), len(Lambdas)), np.nan)
        for i in range(len(results['Lambda'])):
          if results['rescale'][i] == rescale:
            Z[m_labels.index(results['test_fcn_support_radii'][i]), Lambdas.index(results['Lambda'][i])] = results[metric][i]

        if metric == 'R2':
          pcm = ax[j].pcolormesh(Lambdas, range(len(m_labels)), Z, shading='nearest', vmin=0, vmax=1)
        elif metric == 'coeff_err':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, range(len(m_labels)), Z, shading='nearest')
        else:
          pcm = ax[j].pcolormesh(Lambdas, range(len(m_labels)), Z, shading='nearest')
        ax[j].set_xscale('log')
        ax[j].set_yticks(range(len(m_labels)))
        ax[j].set_yticklabels(m_labels)
        ax[j].set_xlabel(r'$\lambda$')
        ax[j].set_ylabel('$test_fcn_support_radii$')
        ax[j].set_title(titles.get(metric, metric))
        fig.colorbar(pcm, ax=ax[j])
      fig.suptitle(f'rescale = {rescale}')
      fig.tight_layout()
      plt.show()
    return
