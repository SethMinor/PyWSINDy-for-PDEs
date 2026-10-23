from helper_fcns_dev import *

# for hyperparam sweeps
import csv
import ast
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

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
                test_fcn_poly_degrees = None, # ....... 1/2 of test fcn poly degrees (list of ints [p_x,...,p_t])
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
      if len(grid) != 1:
        raise ValueError("ODE data requires a single time axis.")
    elif self.U.dim() < 2:
      raise ValueError("PDE data requires at least one spatial axis.")

    # Strings for printing symbolic expressions
    self.var_names = var_names if var_names is not None else ['u']+['v'+str(i+1) for i in range(len(aux_vars))]
    self.grid_axis_names = grid_axis_names if grid_axis_names is not None \
      else [('x','y','z')[i] for i in range(U.ndim-1)] + ['t']
    self.derivative_names = self.get_derivative_names()

    # The discretized space-time grid
    self.grid = grid if type(grid) == list else [grid]
    self.spacing = [xi.diff()[0] for xi in self.grid]
    self.subsample_rate = subsample_rate if subsample_rate is not None else [U.shape[i]//50 for i in range(U.ndim)]

    # Test function parameters
    self.test_fcn_support_radii = test_fcn_support_radii if test_fcn_support_radii is not None \
      else self.compute_test_fcn_support_radii()
    self.test_fcn_poly_degrees = test_fcn_poly_degrees if test_fcn_poly_degrees is not None \
      else self.compute_test_fcn_poly_degrees()

    # Scale-invariant preconditioning
    if rescale:
      [self.x_scales, self.t_scale] = self.compute_spatial_scales()
      self.monomials_max = max([monomial_index_j[0] for monomial_index_j in monomials])
      self.u_scale = self.compute_u_scale(self.U, self.monomials_max)
      self.aux_scales = []
      for i,Vi in enumerate(aux_vars):
        monomials_max = max([monomial_index_j[i+1] for monomial_index_j in self.monomials])
        self.aux_scales.append(self.compute_u_scale(Vi, monomials_max))
    else:
      [self.x_scales, self.t_scale, self.u_scale, self.aux_scales] = 4*[None]

    # Test functions and query points
    [self.mask, self.conv_mask] = self.get_query_points()
    [self.test_fcns, self.test_fcn_kernels] = self.build_test_fcns()

  #################################################################################################################
  # AUTOMATIC HYPERPARAMETER SELECTION AND TUNING
  #################################################################################################################

  def compute_test_fcn_support_radii(self, method='spectral_matching'):
    """
    Automatically selects support radii (i.e., bandwidths) for test functions.
    [Currently, the only method supported is "spectral matching".]
    """
    if method == 'spectral_matching':
      return [self.spectral_matching(d, changepoint, F_root) for d in range(self.U.dim())]
    else:
      raise ValueError(f'Method "{method}" is not supported right now.')

  def compute_test_fcn_poly_degrees(self):
    """
    Automatically selects test function degrees for given support radii.
    [See Equation (4.3) of Messenger, Bortz (JCP, 2021) for details.]
    """
    def compute_degrees(d, m, derivatives, test_fcn_support_tol=1.e-10):
      derivatives_bar = max(tuple(item[d] for item in derivatives))
      log_tau_term = np.ceil(np.log(test_fcn_support_tol)/np.log((2*m-1)/m**2))
      p = int(max(log_tau_term, derivatives_bar + 1))
      return p
    test_fcn_poly_degrees = [compute_degrees(d, md, self.derivatives, test_fcn_support_tol=self.test_fcn_support_tol) \
      for d,md in enumerate(self.test_fcn_support_radii)]
    return test_fcn_poly_degrees

  def spectral_matching(self, d, changepoint, F_root):
    """
    Implements the "spectral matching" method from Appendix A of Messenger, Bortz (JCP, 2021).
    """
    N_d = self.U.shape[d]
    
    Uhat_d = abs(torch.fft.rfft(self.U, n=N_d, dim=d))
    dims = [dim for dim in range(Uhat_d.ndimension()) if dim != d]
    Uhat_d = Uhat_d.mean(dim = dims) if dims else Uhat_d

    cum_sum = torch.cumsum(Uhat_d, dim=0)
    cum_sum = (cum_sum/cum_sum.max()).numpy() # Normalize for curve fitting

    # Solve change-point problem
    freqs = torch.arange(0, np.floor(N_d/2)+1, 1).numpy()
    params = scipy.optimize.curve_fit(changepoint, freqs, cum_sum, p0=self.changepoint_guess)[0]
    k = int(params[0])

    # Solve root-finding problem
    guess = (np.sqrt(3)*N_d*self.test_fcn_spectrum_decay)/(2*np.pi*k)
    guess = guess * (1 + np.sqrt(1 - (8/np.sqrt(3))*np.log(self.test_fcn_support_tol)))/2
    md = int(scipy.optimize.root(F_root, guess, \
      args=(k,N_d,self.test_fcn_spectrum_decay,self.test_fcn_support_tol)).x[0])

    # Optional plotting
    if self.verbosity:
      axis = self.grid_axis_names[d]
      plt.figure(figsize=(7,2))
      Uhat_d = Uhat_d.numpy()
      ylabel = r'Spectrum, $|\mathcal{F}_t[u]|$' if self.equation_type == 'ode' \
        else fr'Spectrum, $|\mathcal{{F}}_{axis}[u]|(k)$'
      plt.plot(freqs, Uhat_d, 'C0.-')
      plt.axvline(params[0], ls='--', color='r', label=fr'Estimated changepoint, $k_{axis}$')
      plt.xlabel('Frequency, $k$', fontsize=10)
      plt.ylabel(ylabel, fontsize=10)
      plt.xscale('log')
      plt.yscale('log')
      plt.title(f'Spectral matching (${axis}$-axis): $m_{axis}={md}$ and $k_{axis}={k}$', fontsize=12)
      plt.legend(loc = 'upper right', fontsize=10, framealpha=0.9)
      plt.grid(True, which='both', color='gray', alpha=0.2)
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
      max_d = max(tuple(deriv_index_i[d] for deriv_index_i in self.derivatives))
      max_x.append(max(1,max_d))
    max_t = max(1,max(tuple(deriv_index_i[-1] for deriv_index_i in self.derivatives)))

    # Ansatz given in Equation (4.9) of Messenger, Bortz (JCP, 2021)
    x_scales = [(1/(self.test_fcn_support_radii[d]*self.spacing[d])
      * (my_nchoosek(self.test_fcn_poly_degrees[d], max_x[d]/2)
      * factorial(max_x[d]))**(1/max_x[d])).item() for d in range(D)]
    t_scale = (1 / (self.test_fcn_support_radii[-1] * self.spacing[-1])
      * (my_nchoosek(self.test_fcn_poly_degrees[-1], max_t/2)
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
      monomial_index_j = powers[j]
      if (aj==None) or (monomial_index_j==None):
        mu[j] = 1.
      else:
        x_scale_exps = [x_scales[d]**(self.derivatives[0][d] - self.derivatives[aj][d]) for d in range(D)]
        x_scale_term = np.prod(x_scale_exps)
        t_scale_term = t_scale**(self.derivatives[0][-1] - self.derivatives[aj][-1])

        u_scale_term = [u_scale[n]**(monomial_index_j[n]) for n in range(len(u_scale))]
        u_scale_term = np.prod(u_scale_term)

        mu[j] = (u_scale_term * x_scale_term * t_scale_term)/self.u_scale
    return mu

  #################################################################################################################
  # TEST FUNCTION (AND QUERY POINT) CONSTRUCTION
  #################################################################################################################

  def get_query_points(self):
    """
    Returns query point indices in two forms:
    --> "mask" is defined for Nx x ... x Nt tensors and ignores padding (i.e., U[mask][k] = U[x_k,...,t_k])
    --> "conv_mask" is defined relative to convolution padding (i.e., G[conv_mask][k] = G[x_k-m_x,...,t_k-m_t])
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

        plt.figure(figsize=(7,2))
        plt.axhline(0, color='gray', alpha=0.3)
        plt.plot(t, self.U, 'C0.-', alpha=0.8)
        plt.plot(tk, self.U[mask], 'r.', ms=6, label='Query points')
        plt.xlabel('Time, $t$')
        plt.ylabel('$' + self.var_names[0] + '(t)$')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, color='gray', alpha=0.2)
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

  def build_1D_test_fcns(self, d):
    """
    Constructs the 1D component (and its derivatives) of the separable test function along the d-th axis.
    --> Returns a tensor "test_fcns_d" of shape: [number of derivatives, 2*support_radius + 1].
    --> Note: "test_fcns_d[i,:]" indexes the i-th derivative.
    """
    [x, support_radius, poly_degree] = \
      [self.grid[d], self.test_fcn_support_radii[d],self.test_fcn_poly_degrees[d]]
    dx = (x[1] - x[0]).item()

    if (support_radius > (len(x)-1)/2) or (support_radius <= 1):
      raise ValueError('Error: invalid test function support.')

    if self.rescale:
      scale_d = (self.x_scales + [self.t_scale])[d]
      dx *= scale_d

    # Initialize grid of discretized test fcn values
    test_fcns_d = torch.zeros(len(self.derivatives), 2*support_radius+1, dtype=torch.float64)
    support_grid_d = torch.linspace(-1, 1, 2*support_radius+1, dtype=torch.float64)
    multi_index_d = tuple(deriv_index_i[d] for deriv_index_i in self.derivatives)

    x_sym = sp.Symbol('x')
    phi_bar = (1 - x_sym**2)**poly_degree
    vec = np.vectorize(D_phibar)

    for i in range(len(self.derivatives)):
      if (i > 0) and (multi_index_d[i-1] == multi_index_d[i]):
        test_fcns_d[i,:] += test_fcns_d[i-1,:]
      else:
        num_derivs = multi_index_d[i]
        A_i = torch.from_numpy(vec(support_grid_d, num_derivs, x_sym, phi_bar))
        test_fcns_d[i,:] += (1/((support_radius*dx)**num_derivs)) * A_i

    # Optional plotting
    if self.verbosity:
      plt.figure(figsize=(7,2))
      plt.axhline(0, color='gray', alpha=0.3)
      for i in range(len(test_fcns_d[:,0])):
        axis = self.grid_axis_names[d]

        if i == 0:
          label = fr'$\phi_{axis}({axis}_k - {axis})$'
          plt.plot(support_radius*dx*support_grid_d, test_fcns_d[i,:], '-', label=label)
        
        # Check if test function is the same as the previous one
        elif i <= 4 and multi_index_d[i] != multi_index_d[i-1]:
          if i < 4:
            label = fr'$\frac{{d^{i}}}{{d{axis}^{i}}} \phi_{axis}({axis}_k - {axis})$'
          else:
            label = r'$\dots$'
          plt.plot(support_radius*dx*support_grid_d, test_fcns_d[i,:], '--', label=label)
        
        # After four test functions, the legend get too crowded
        elif i > 4 and multi_index_d[i] != multi_index_d[i-1]:
          plt.plot(support_radius*dx*support_grid_d, test_fcns_d[i,:], '--')
        
      plt.title(f'Test functions (${axis}$-axis)')
      plt.xlabel(fr'${axis}_k - {axis}$')
      plt.grid(True, color='gray', alpha=0.2)
      plt.legend(loc='upper right', fontsize=8, framealpha=0.5)
      plt.show()
    return test_fcns_d

  def build_test_fcns(self):
    """
    Builds the separable test function and its derivatives.
    Returns:
    --> "test_fcns" = Kronecker product of 1D test fcn components along the x,...,t axes, arranged as a
                      list of "num_derivs" tensors of shape (2*support_radii[0]+1,...,2*support_radii[D]+1)
    --> "test_fcn_kernels" = a list of the 1D kernels corresponding to each "test_fcns" tensor
    """
    D = len(self.grid)
    num_derivs = len(self.derivatives)

    # Build all 1D test functions (along x,...,t axes)
    test_fcn_axes = [self.build_1D_test_fcns(d) for d in range(D)]
    test_fcn_kernels = [[test_fcn_axes[d][i,:] for d in range(D)] for i in range(num_derivs)]
    test_fcns = []

    # Kronecker product along axes
    for i in range(num_derivs):
      test_fcn_kernel = test_fcn_kernels[i]
      ijk = [chr(105 + d) for d in range(D)] # ["i", "j", "k", ...]
      einsum = ','.join(ijk) + '->' + ''.join(ijk)
      D_phi = torch.einsum(einsum, *test_fcn_kernel)
      test_fcns.append(D_phi)
    
    # Optional heatmap plotting
    if self.verbosity and self.equation_type == 'pde':

      # All unique derivative combos (along the first two axes)
      combos = {tuple(deriv_index_i[:2]): i for \
        i,deriv_index_i in sorted(enumerate(self.derivatives), key=lambda p: not any(p[1][2:]))}
      [rows, cols] = [-(-len(combos)//4), min(4, len(combos))]
      axes = ','.join(self.grid_axis_names[:2]) + (r',\dots' if D > 2 else '')

      plt.figure(figsize=(3*cols, 3*rows))
      for n,i in enumerate(combos.values()):
        psi = torch.outer(*test_fcn_kernels[i][:2]).T
        name = self.derivative_names[i]
        plt.subplot(rows, cols, n+1)
        plt.pcolormesh(psi, cmap='coolwarm', vmin=-psi.abs().max(), vmax=psi.abs().max())
        plt.title('$' + (r'\partial' + name if name else '') + fr'\psi({axes})$')
        plt.axis('off')
      plt.tight_layout()
      plt.show()
    return test_fcns, test_fcn_kernels

  #################################################################################################################
  # CONSTRUCTION OF WSINDY LINEAR SYSTEM
  #################################################################################################################

  def build_lhs(self):
    kernel = self.test_fcn_kernels[0]
    if self.rescale:
      scales_product = np.prod(self.x_scales + [self.t_scale])
      lhs = compute_weak_poly(self.U, kernel, self.spacing, \
        u_scale=self.u_scale, scales_product=scales_product, volume_element=self.volume_element)
    else:
      lhs = compute_weak_poly(self.U, kernel, self.spacing, volume_element=self.volume_element)
    b = lhs[self.conv_mask]
    self.lhs_name = self.format_lhs_name()
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

    for i,deriv_index_i in enumerate(tqdm(self.derivatives[1:]), start=1):
      kernel = self.test_fcn_kernels[i]
      for j,monomial_index_j in enumerate(self.monomials):
        if all(monomial_index_j_d==0 for monomial_index_j_d in monomial_index_j) \
          and any(deriv_index_i_d!=0 for deriv_index_i_d in deriv_index_i):
          continue # Avoid derivatives of constant terms
        else:
          assert (len(monomial_index_j)-1) == len(self.aux_vars), "Inconsistent number of powers!"
          derivs.append(i)
          powers.append(monomial_index_j)
          term = compute_weak_multipoly(state, kernel, self.spacing, power=monomial_index_j, u_scale=u_scale, \
            scales_product=scales_product, volume_element=self.volume_element)
          name = self.format_monomial(monomial_index_j) + self.derivative_names[i]
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
  
  #################################################################################################################
  # FORMATTING SYMBOLIC STRINGS AND PRINTING RESULTS
  #################################################################################################################
  
  def get_derivative_names(self):
    """
    Returns a list of all symbolic derivative names used in the library.
    """
    spatial_dimension = self.U.dim() - 1
    derivative_names = []
    for elem in self.derivatives:
      if all(value == 0 for value in elem):
        derivative_names.append('')
      else:
        names = self.grid_axis_names
        order = [spatial_dimension] + list(range(spatial_dimension)) # t,x,y,z,... order
        derivative_names.append('_{' + ''.join(names[d]*elem[d] for d in order) + '}')
    return derivative_names

  def format_lhs_name(self):
    """
    Symbolic string for the LHS evolution operator (dot notation for ODEs).
    """
    name = self.var_names[0]
    order = self.derivatives[0][-1] # time derivative order
    if self.equation_type == 'ode' and order > 0:
      dots = {1: r'\dot', 2: r'\ddot', 3: r'\dddot', 4: r'\ddddot'}
      return f'{dots[order]}{{{name}}}' if order in dots else f'{name}^{{({order})}}'
    return name + self.derivative_names[0]

  def format_monomial(self, monomial_index_j):
    """
    Symbolic string formatting for monomial terms in the library.
    --> "monomial_index_j" = multi-index of the j-th monomial term in "self.monomials"
    """
    terms = []
    for d in range(len(monomial_index_j)):
      if monomial_index_j[d] == 0:
        continue
      if monomial_index_j[d] == 1:
        terms.append(self.var_names[d])
      else:
        terms.append(f'{self.var_names[d]}^{monomial_index_j[d]}')
    return '(' + ' '.join(terms) + ')' if terms else '(1)'

  def print_report(self):
    """
    Prints a summary/report of the WSINDy run.
    """
    print('HYPER-PARAMETERS')
    print(f'test_fcn_support_radii = {self.test_fcn_support_radii}')
    print(f'test_fcn_poly_degrees = {self.test_fcn_poly_degrees}')
    print(f'subsample_rate = {self.subsample_rate}')
    if self.rescale:
      if self.equation_type == 'ode':
        print(f't_scale = {self.t_scale:.2f}')
      else:
        scales = self.x_scales + [self.t_scale]
        print('[x_scales, t_scale] = ' + str([float(f" {yi:.2f}") for yi in scales]))
      print(f'u_scale = {self.u_scale:.2f}')
      aux_scales = [float(f'{scale:.2f}') for scale in self.aux_scales]
      print(f'aux_scales = {aux_scales}\n')
    else:
      print('Not rescaled.\n')

    print('LIBRARY')
    print(f'Number of query points = {self.library.shape[0]}')
    print(f'Number of candidate terms = {self.library.shape[1]}')
    print(f'cond(G) = {la.cond(self.library):.2e}\n')

    print('RESULTS')
    eq_label = self.equation_type.upper()
    eqn = symbolic_eqn(self.lhs_name, self.rhs_names, self.coeffs)
    num_terms = self.coeffs.count_nonzero().item()
    if self.rescale:
      [r, R2] = compute_residuals(self.library, self.coeffs/self.mu, self.lhs)
    else:
      [r, R2] = compute_residuals(self.library, self.coeffs, self.lhs)
    print(f'{eq_label}: {eqn}')
    print(f'R^2 = {R2:.2f}')
    print(f'Nonzero terms = {num_terms}')
    print(f'Lambda = {self.Lambda:.2e}')
    print(f'Loss = {self.loss:.2f}')
    print(f'Relative L2 error = {la.norm(r)/la.norm(self.lhs):.2f}')
    return

  def help_attributes(self):
    """
    Prints a help message with all the class attributes and their descriptions.
    """

  def help_methods(self):
    """
    Prints a help message with all the class methods and their descriptions.
    """

  def help(self):
    """
    Calls both "help_attributes" and "help_methods".
    """

  #################################################################################################################
  # HYPERPARAMETER SWEEPS
  #################################################################################################################

  def hyperparameter_sweep(self, lhs_name=None, support_radii=None, Lambdas=None, rescales=(True, False),
                                 library_fcn=None, true_coeffs=None, csv_path=None, plot=True, noise=None):
    """
    Sweeps over hyperparameters (test_fcn_support_radii, Lambda, rescaling on/off) and reports the results.
    """
    # Set evolution operator name
    if lhs_name is None:
      lhs_name = getattr(self, 'lhs_name', self.format_lhs_name())
    
    # Set support radii to sweep over
    if support_radii is None:
      support_radii = []
      for mult_factor in np.linspace(0.5, 1.5, 60):
        radii = [int(round(mult_factor*mi)) for mi in self.test_fcn_support_radii]
        radii = [min(max(mi,2), (self.U.shape[i]-1)//2) for i,mi in enumerate(radii)]
        if radii not in support_radii:
          support_radii.append(radii)
    else:
      support_radii = [(self.U.ndim*[mi] if type(mi)==int else list(mi)) for mi in support_radii]
    
    # Set MSTLS thresholds to sweep over
    if Lambdas is None:
      #Lambdas = 10**((4/49)*torch.arange(0,50)-4)
      Lambdas = 10**((3/199)*torch.arange(0,200)-3)
    Lambdas = [float(Li) for Li in Lambdas]

    # Metrics to track during the sweep
    metrics = ['test_fcn_support_radii', 'test_fcn_poly_degrees', 'Lambda', 'rescale', 'model',
               'terms', 'coeffs', 'sparsity', 'loss', 'R2', 'L2', 'rel_L2', 'cond_G', 'noise']
    if true_coeffs is not None:
      metrics.extend(['coeff_error','support_error','rel_sparsity'])

    results = {key: [] for key in metrics}
    subsample_rate = self.U.ndim * [1]

    # Perform the actual sweep
    for test_fcn_support_radii in tqdm(support_radii):
      for rescale in rescales:
        model = WSINDy(
          self.U,
          self.derivatives,
          self.monomials,
          self.grid,
          aux_vars=self.aux_vars,
          var_names=self.var_names,
          test_fcn_support_radii=test_fcn_support_radii,
          subsample_rate=subsample_rate,
          volume_element=self.volume_element,
          test_fcn_support_tol=self.test_fcn_support_tol,
          test_fcn_spectrum_decay=self.test_fcn_spectrum_decay,
          changepoint_guess=self.changepoint_guess,
          verbosity=False,
          rescale=rescale,
          equation_type=self.equation_type
        )
        if library_fcn is None:
          [G, powers, derivs, rhs_names] = model.create_default_library()
        else:
          [G, powers, derivs, rhs_names] = library_fcn(model)
        model.build_lhs()
        model.set_library(G, powers, derivs, rhs_names)
        cond_G = la.cond(model.library).item()

        for Lambda in Lambdas:
          w = model.MSTLS(Lambda=Lambda)
          w_tilde = w/model.mu if rescale else w
          [r, R2] = compute_residuals(model.library, w_tilde, model.lhs)
          nonzero = w.nonzero().flatten().tolist()

          results['test_fcn_support_radii'].append(str(test_fcn_support_radii))
          results['test_fcn_poly_degrees'].append(str(model.test_fcn_poly_degrees))
          results['Lambda'].append(Lambda)
          results['rescale'].append(rescale)
          results['model'].append(symbolic_eqn(lhs_name, rhs_names, w))
          results['terms'].append([rhs_names[j] for j in nonzero])
          results['coeffs'].append([w[j].item() for j in nonzero])
          results['sparsity'].append(len(nonzero))
          results['loss'].append(model.loss)
          results['R2'].append(R2.item())
          results['L2'].append(la.norm(r).item())
          results['rel_L2'].append((la.norm(r)/la.norm(model.lhs)).item())
          results['cond_G'].append(cond_G)

          if noise is not None:
            results['noise'].append(noise)
          else:
            results['noise'].append('unknown')

          # Maximum relative coefficient error
          if true_coeffs is not None:
            learned = dict(zip(results['terms'][-1], results['coeffs'][-1]))
            error = max(abs(learned.get(name,0.) - wj)/abs(wj) for name,wj in true_coeffs.items())
            results['coeff_error'].append(error)
            true_support = {name for name,wj in true_coeffs.items() if wj != 0}
            results['support_error'].append(len(set(learned) ^ true_support))
            results['rel_sparsity'].append(len(nonzero) - len(true_support))

    if csv_path is not None:
      with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics)
        for i in range(len(results['Lambda'])):
          writer.writerow([results[key][i] for key in metrics])

    if plot:
      self.plot_sweep_results(results)
    return results

  def plot_sweep_results(self, results, metrics=None, noise=None):
    """
    Plot heatmaps of hyperparameter sweep results over the (test_fcn_support_radii, Lambda) plane.
    """
    if metrics is None:
      metrics = ('loss','L2','sparsity')
      metrics += tuple(metric for metric in ('coeff_error','support_error','rel_sparsity') if metric in results)

    radii_labels = list(dict.fromkeys(results['m']))
    Lambdas = list(dict.fromkeys(results['Lambda']))
    rescales = list(dict.fromkeys(results['rescale']))
    [radii_ticks, tick_labels, order] = parse_m_labels(radii_labels)
    titles = {'loss': r'MSTLS loss, $\log_{10}\mathcal{L}(\mathbf{w}^{\lambda})$',
              'R2': r'Equation fit, $R^2$',
              'L2': r'$L^2$ error, $\log_{10}\|\mathbf{Gw}^{\lambda}-\mathbf{b}\|_2$',
              'sparsity': r'Nonzero terms, $\|\mathbf{w}^{\lambda}\|_0$',
              'coeff_error': r'Coeff. error, $\log_{10}E_{\infty}$',
              'support_error': 'Support error (# indices)',
              'rel_sparsity': r'Relative sparsity, $\|\mathbf{w}^{\lambda}\|_0-\|\mathbf{w}^{\star}\|_0$'}
    
    # Some helper functions for plotting
    def parse_m_labels(radii_labels):
      try:
        radii = [ast.literal_eval(label) for label in radii_labels]
      except (SyntaxError, ValueError):
        radii = None
      if (radii is not None) and all(len(set(np.atleast_1d(mi)))==1 for mi in radii):
        radii_ticks = np.array([np.atleast_1d(mi)[0] for mi in radii])
        order = np.argsort(radii_ticks)
        return radii_ticks[order], None, order
      return np.arange(len(radii_labels)), radii_labels, np.arange(len(radii_labels))
    
    def sweep_grid(results, metric, rescale, radii_labels, Lambdas, order):
      Z = np.full((len(radii_labels), len(Lambdas)), np.nan)
      for i in range(len(results['Lambda'])):
        if results['rescale'][i] == rescale:
          Z[radii_labels.index(results['test_fcn_support_radii'][i]), Lambdas.index(results['Lambda'][i])] = results[metric][i]
      return Z[order]

    def pad_ticks(ticks, positive=False):
      if len(ticks) > 1:
        left, right = 2*ticks[0]-ticks[1], 2*ticks[-1]-ticks[-2]
      else:
        step = ticks[0]/2 if positive else 1
        left, right = ticks[0]-step, ticks[0]+step
      if positive and left <= 0:
        left = ticks[0]/2
      return np.r_[left, ticks, right]

    # PLOT THIS PUPPY
    for rescale in rescales:
      nrows = 2 if len(metrics) > 4 else 1
      ncols = int(np.ceil(len(metrics)/nrows))
      fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols,4*nrows), sharex=True, sharey=True)
      ax = np.atleast_1d(ax).flatten()

      support_Z = sweep_grid(results, 'support_error', rescale, radii_labels, Lambdas, order) \
                  if 'support_error' in results else None
      correct = support_Z == 0 if support_Z is not None else None
      coeff_Z = sweep_grid(results, 'coeff_error', rescale, radii_labels, Lambdas, order) \
                if 'coeff_error' in results else None
      finite_coeffs = np.isfinite(coeff_Z) & correct if coeff_Z is not None and correct is not None else None
      best = None
      if finite_coeffs is not None and finite_coeffs.any():
        best = np.unravel_index(np.nanargmin(np.where(finite_coeffs, coeff_Z, np.nan)), coeff_Z.shape)

      for j,metric in enumerate(metrics):
        Z = sweep_grid(results, metric, rescale, radii_labels, Lambdas, order)
        cbar_ticks = None

        if metric == 'loss':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, radii_ticks, Z, cmap='coolwarm')
        elif metric == 'R2':
          pcm = ax[j].pcolormesh(Lambdas, radii_ticks, Z, cmap='Reds_r', vmin=0, vmax=1)
        elif metric == 'L2':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, radii_ticks, Z, cmap='coolwarm')
        elif metric in ('sparsity','support_error'):
          vmax = int(np.nanmax(Z))
          cbar_ticks = np.arange(vmax + 1)
          bounds = np.arange(-0.5, vmax + 1.5)
          norm = BoundaryNorm(bounds, plt.get_cmap('bone_r').N)
          pcm = ax[j].pcolormesh(Lambdas, radii_ticks, Z, cmap='bone_r', norm=norm)
        elif metric == 'rel_sparsity':
          vmax = int(np.nanmax(np.abs(Z)))
          cbar_ticks = np.arange(-vmax, vmax + 1)
          bounds = np.arange(-vmax - 0.5, vmax + 1.5)
          norm = BoundaryNorm(bounds, plt.get_cmap('coolwarm').N)
          pcm = ax[j].pcolormesh(Lambdas, radii_ticks, Z, cmap='PiYG', norm=norm)
        elif metric == 'coeff_error':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, radii_ticks, Z, cmap='coolwarm')
        else:
          pcm = ax[j].pcolormesh(Lambdas, radii_ticks, Z, cmap='Reds')

        if correct is not None and correct.any():
          ax[j].contour(pad_ticks(Lambdas, positive=True), pad_ticks(radii_ticks),
                        np.pad(correct.astype(float), 1), levels=[0.5],
                        colors='darkgoldenrod', linestyles='--', linewidths=1, zorder=3)
        if best is not None:
          ax[j].plot(Lambdas[best[1]], radii_ticks[best[0]], marker='*', color='darkgoldenrod',
                     markersize=7, linestyle='none', zorder=4)

        ax[j].set_xscale('log')
        if tick_labels is not None:
          ax[j].set_yticks(radii_ticks)
          ax[j].set_yticklabels(tick_labels, fontsize=7)
        ax[j].set_xlabel(r'$\lambda$')
        ax[j].set_ylabel('$m$')
        ax[j].set_title(titles.get(metric, metric))
        fig.colorbar(pcm, ax=ax[j], ticks=cbar_ticks)
      for extra_ax in ax[len(metrics):]:
        extra_ax.set_visible(False)
      fig.suptitle(f'Rescaled: {rescale}, Noise: {noise}')
      legend_handles = ([Line2D([0],[0], color='darkgoldenrod', linestyle='--', label='Correct support')]
                        if support_Z is not None else [])
      if best is not None:
        legend_handles.append(Line2D([0],[0], color='darkgoldenrod', marker='*', linestyle='none',
                                     markersize=7, label='Min. coeff. error (in support)'))
      if legend_handles:
        fig.legend(handles=legend_handles, loc='upper right', ncol=len(legend_handles))
      fig.tight_layout(rect=(0,0,1,0.9) if legend_handles else None)
      plt.show()
    return
