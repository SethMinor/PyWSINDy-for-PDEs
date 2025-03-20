class WSINDy:
  def __init__(self, U, alpha, X, m=None, p=None, s=None, tau=1e-10, tau_hat=2,
               verbosity=True, init_guess=[10,1,10,0], rescale=True, beta_max=1, aux_scales=[]):
    self.alpha = alpha
    self.X = X
    self.spacing = [xi.diff()[0] for xi in X]
    self.U = U
    self.tau = tau
    self.tau_hat = tau_hat
    self.verbosity = verbosity
    self.init_guess = init_guess
    self.rescale = rescale
    self.beta_max = beta_max
    self.aux_scales = aux_scales

    if m is None:
      self.m = self.compute_m()
    else:
      self.m = m

    if p is None:
      self.p = self.compute_p()
    else:
      self.p = p

    if s is None:
      self.s = [U.shape[i]//50 for i in range(U.dim())]
    else:
      self.s = s

    if rescale:
      [self.yx, self.yt] = self.compute_spatial_scales()
      self.yu = compute_u_scale(self.U, self.beta_max)
    else:
      [self.yx, self.yt, self.yu] = [None, None, None]

    self.mask = self.compute_query_points()
    self.axes = self.build_axes()
    self.test_fcns = self.build_test_fcns()
    self.derivative_names = self.get_derivative_names()

  def compute_m(self):
    m = [self.optimal_support(d, changepoint, F_root) for d in range(self.U.dim())]
    return m

  def compute_p(self):
    p = [compute_degrees(d, md, self.alpha, tau=self.tau) for d,md in enumerate(self.m)]
    return p

  def optimal_support(self, d, changepoint, F_root):
    Nd = self.U.shape[d]
    Uhat_d = abs(torch.fft.rfft(self.U, n=Nd, dim=d))
    dims = [dim for dim in range(Uhat_d.ndimension()) if dim != d]
    Uhat_d = Uhat_d.mean(dim = dims)

    Hd = torch.cumsum(Uhat_d, dim=0)
    Hd = (Hd/Hd.max()).numpy() # Normalize for curve fitting

    # Solve change-point problem
    freqs = torch.arange(0, np.floor(Nd/2)+1, 1).numpy()
    params = scipy.optimize.curve_fit(changepoint, freqs, Hd, p0=self.init_guess)[0]
    k = int(params[0])

    # Solve root-finding problem
    guess = (np.sqrt(3)*Nd*self.tau_hat)/(2*np.pi*k)
    guess = guess * (1 + np.sqrt(1 - (8/np.sqrt(3))*np.log(self.tau)))/2
    md = int(scipy.optimize.root(F_root, guess, args=(k,Nd,self.tau_hat,self.tau)).x[0])

    if self.verbosity:
      plt.figure(figsize=(6,3))
      Uhat_d = Uhat_d.numpy()
      plt.plot(freqs, Uhat_d, 'r--', label='Spectrum, $\mathcal{F}_d[U]$')
      plt.plot(freqs, Uhat_d.max()*Hd, 'k', label='Cum. Sum, $H_d$')
      plt.plot(freqs, Uhat_d.max()*changepoint(freqs,*params), 'g--')
      plt.plot(params[0], Uhat_d.max()*params[1], 'go', label='Changepoint, $k_d$')
      plt.xlabel('$k$')
      plt.title(f'Axis $d=${d}: $m_d={md}$ and $k_d=${k}')
      plt.legend(loc = 'upper right')
      plt.grid(True)
      plt.show()
    return md

  # Compute mask of query point indices, U[mask] = U[xk,...,tk]
  def compute_query_points(self):
    subsamples = [subsample(self.s[i], self.m[i], self.X[i]) for i in range(self.U.dim())]
    cartesian_prod = itertools.product(*subsamples)
    mask = tuple(map(torch.tensor, zip(*cartesian_prod)))

    if self.verbosity:
      [x1, x2] = [X[0][subsamples[0]], X[1][subsamples[1]]]
      [X1, X2] = np.meshgrid(x1, x2)
      UK = reshape_subsampled_tensor(self.U, mask, self.s, self.m, self.X)
      inds = (slice(None),slice(None)) + (self.U.dim()-2)*(0,)

      plt.figure(figsize=(4,4))
      plt.pcolormesh(X1, X2, UK[inds].T, cmap='coolwarm')
      plt.axis('equal')
      plt.xlabel('$x_1$')
      plt.ylabel('$x_2$')
      plt.title('Subsampled Data (Slice 0)')
      plt.show()
    return mask

  # Compute spatio-temporal scales yx, yt
  def compute_spatial_scales(self):
    D = len(self.spacing)-1

    max_x = []
    for d in range(len(alpha[0]) - 1):
      max_d = max(tuple(item[d] for item in alpha))
      max_x.append(max_d)
    max_t = max(tuple(item[-1] for item in alpha))

    # Ansatz given in the paper
    yx = [(1/(self.m[d] * self.spacing[d]) * (my_nchoosek(self.p[d], max_x[d]/2)
          * scipy.special.factorial(max_x[d]))**(1/max_x[d])).item() for d in range(D)]
    yt = (1 / (self.m[-1] * self.spacing[-1]) * (my_nchoosek(self.p[-1], max_t/2)
          * scipy.special.factorial(max_t))**(1/max_t)).item()
    return yx,yt

  # Compute scale matrix diagonal, M = diag(mu)
  # For each term D^i[f_j(u)]:
  #  - beta = [j's]
  #  - derivs = [i's]
  def compute_scale_matrix(self, beta, derivs):
    D = self.U.dim() - 1
    [yx, yt] = [self.yx, self.yt]
    yu = [self.yu] + self.aux_scales

    num_terms = len(beta)
    mu = torch.zeros(num_terms, dtype=torch.float64)
    for j in range(num_terms):
      i = derivs[j]
      yx_exps = [yx[d]**(self.alpha[0][d] - self.alpha[i][d]) for d in range(D)]
      yx_term = np.prod(yx_exps)
      yt_term = yt**(self.alpha[0][-1] - self.alpha[i][-1])

      yu_term = [yu[n]**(beta[j][n]) for n in range(len(yu))]
      yu_term = np.prod(yu_term)

      mu[j] = yu_term * yx_term * yt_term
    return mu/self.yu

  # Returns symbolic derivatives
  def get_derivative_names(self):
    D = self.U.dim() - 1
    derivative_names = []
    for elem in self.alpha:
      if all(value == 0 for value in elem):
        derivative_names.append('')
      else:
        # (1+1)-D, (2+1)-D, (3+1)-D case-handling
        if D == 1:
          derivative_names.append('_'+'t'*elem[1]+'x'*elem[0])
        elif D == 2:
          derivative_names.append('_'+'t'*elem[2]+'x'*elem[0]+'y'*elem[1])
        elif D == 3:
          derivative_names.append('_'+'t'*elem[3]+'x'*elem[0]+'y'*elem[1]+'z'*elem[2])
    return derivative_names

  # Compute separable component along d-th axis
  def get_test_fcns(self, d):
    x = self.X[d]
    Nd = len(x)
    dx = (x[1] - x[0]).item()
    m = self.m[d]
    p = self.p[d]

    if (m > (Nd-1)/2) or (m <= 1):
      raise ValueError('Error: invalid test function support.')

    if self.rescale:
      scale_d = (self.yx + [self.yt])[d]
      dx *= scale_d

    # Initialize grid of discretized test fcn values
    test_fcns_d = torch.zeros(len(self.alpha), 2*m+1, dtype=torch.float64)
    n_grid = torch.linspace(-1, 1, 2*m+1, dtype=torch.float64)
    multi_index_d = tuple(item[d] for item in self.alpha)

    x_sym = sp.Symbol('x')
    phi_bar = (1 - x_sym**2)**p
    vec = np.vectorize(D_phibar)

    for i in range(len(self.alpha)):
      if (i > 0) and (multi_index_d[i-1] == multi_index_d[i]):
        test_fcns_d[i,:] += test_fcns_d[i-1,:]
      else:
        # Evaluate the (a_d^i)-th derivative
        num_derivs = multi_index_d[i]
        A_i = torch.from_numpy(vec(n_grid, num_derivs, x_sym, phi_bar))
        test_fcns_d[i,:] += (1/((m*dx)**num_derivs)) * A_i

    if self.verbosity:
      plt.figure(figsize=(6,3))
      for i in range(len(test_fcns_d[:,0])):
        plt.plot(m*dx*n_grid, test_fcns_d[i,:], '--.', label=f'$i={i}$')
      plt.title(f'Axis $d={d}$ Test Functions')
      plt.xlabel('$x_k - x$')
      plt.ylabel('$\mathcal{D}^i\phi_d(x_k - x)$')
      plt.grid(True)
      plt.legend(loc='upper right')
      plt.show()
    return test_fcns_d

  def build_axes(self):
    return [self.get_test_fcns(d) for d in range(len(self.m))]

  # def build_test_fcns(self):
  #   num_derivs = len(self.axes[0])
  #   D = len(self.axes)
  #   TEST_FCNS = []
  #   for s in range(num_derivs):
  #       components = [self.axes[d][s,:] for d in range(D)]
  #       result = torch.einsum('i,j->ij', *components) if D == 2 else torch.einsum('i,j,k->ijk', *components)
  #       TEST_FCNS.append(result)
  #   return TEST_FCNS
  def build_test_fcns(self):
    num_derivs = len(self.axes[0])
    D = len(self.axes)
    TEST_FCNS = []
    for s in range(num_derivs):
      components = [self.axes[d][s, :] for d in range(D)]
      einsum_str = ','.join(chr(105 + i) for i in range(D)) + '->' + ''.join(chr(105 + i) for i in range(D))
      result = torch.einsum(einsum_str, *components)
      TEST_FCNS.append(result)
    return TEST_FCNS

  def build_lhs(self, lhs_name):
    yxyt = np.prod(self.yx+[self.yt])
    lhs = compute_weak_dudt(self.U, self.test_fcns[0], self.spacing, yu=self.yu, yxyt=yxyt)
    b = lhs[self.mask]
    self.lhs_name = lhs_name
    self.lhs = b
    return

  def set_library(self, G, rhs_names, beta=None, derivs=None):
    if G.shape[0] != len(self.mask[0]):
      raise ValueError("Library has inconsistent dimensions.")
    self.rhs_names = rhs_names
    self.library = G
    if self.rescale:
      if (beta is None) or (derivs is None):
        raise ValueError("'Beta' and 'derivs' required for scaling matrix.")
      self.mu = self.compute_scale_matrix(beta, derivs)
    return

  # Full MSTLS optimization routine
  def MSTLS(self, lambdas=None, threshold=None):
    w_LS = la.lstsq(self.library, self.lhs, driver='gelsd').solution

    if lambdas is None:
      lambdas = 10**((4/49)*torch.arange(0,50)-4)

    # Check if known optimal threshold was provided
    if threshold is not None:
      lambda_star = threshold

    # Otherwise, iterate to find the optimal theshold 'lambda_star'
    else:
      loss_history = []

      for lambda_n in lambdas:
        w_n, loss_n = self.MSTLS_iterate(lambda_n.item(), w_LS=w_LS.clone())
        loss_history.append(loss_n)

      # Find optimal candidate threshold (smallest minimzer, if not unique)
      ind_star = loss_history.index(min(loss_history))
      lambda_star = lambdas[ind_star].item()

    # Return final result
    w_star,loss_star = self.MSTLS_iterate(lambda_star, w_LS=w_LS.clone())
    if self.rescale:
      w_star = self.mu * w_star
    self.lambda_star = lambda_star
    self.loss_star = loss_star
    self.weights = w_star
    return w_star

  # MSTLS inner loop
  def MSTLS_iterate(self, lambda_n, w_LS=None):
    G = self.library
    b = self.lhs
    max_its = G.shape[1]

    if w_LS is None:
      w_LS = la.lstsq(G, b, driver='gelsd').solution

    # Compute |b|/|Gi| bounds for all columns
    norm_b = la.norm(b)
    norm_Gi = la.norm(G,dim=0)
    bounds = norm_b / norm_Gi

    # Define upper and lower bounds (lambda thresholding)
    L_bounds = lambda_n * torch.maximum(bounds, torch.ones(bounds.shape[0]))
    U_bounds = (1/lambda_n) * torch.minimum(bounds, torch.ones(bounds.shape[0]))

    # Apply iterative thresholding on weight vector
    iteration = 0
    w_n = w_LS.clone()
    inds_old = torch.tensor([0])
    while iteration <= max_its:

      # Find in-bound and out-of-bound indices and set them to zero
      ib_inds = torch.where((abs(w_n) >= L_bounds) & (abs(w_n) <= U_bounds))[0]
      oob_inds = torch.where((abs(w_n) < L_bounds) | (abs(w_n) > U_bounds))[0]

      if (torch.equal(inds_old, ib_inds) and iteration!=0) or (ib_inds.shape[0]==0):
        break

      # Find LS solution amongst sparser, in-bound indices
      w_n[ib_inds] = la.lstsq(G[:,ib_inds], b, driver='gelsd').solution
      w_n[oob_inds] = 0

      inds_old = ib_inds
      iteration += 1
      if iteration == max_its:
        print('MSTLS reached the maximum number of iterations allowed.')

    # Evaluate the loss function on the resulting weights
    loss_n = loss(w_n, w_LS, G)
    return w_n, loss_n

  # Prints a report of the WSINDy run
  def print_report(self):
    print('HYPER-PARAMETERS')
    print(f'm = {self.m}')
    print(f'p = {self.p}')
    print(f's = {self.s}')
    if self.rescale:
      scales = self.yx + [self.yt]
      print('[yx, yt] = ' + str([float(f" {yi:.3f}") for yi in scales]))
      print(f'yu = {self.yu:.3f}\n')
    else:
      print('Not rescaled.\n')

    print('LIBRARY')
    print(f'Num. query points = {self.library.shape[0]}')
    print(f'Num. terms = {self.library.shape[1]}')
    print(f'cond(G) = {la.cond(self.library):.3e}\n')

    print('RESULTS')
    pde = symbolic_pde(self.lhs_name, self.rhs_names, self.weights)
    [r, R2] = compute_residuals(self.library, self.weights/self.mu, self.lhs)
    print(f'PDE: {pde}')
    print(f'Relative L2 error = {la.norm(r)/la.norm(self.lhs):.3f}')
    print(f'R^2 = {R2:.3f}')
    print(f'Threshold = {self.lambda_star:.3e}')
    print(f'Loss = {self.loss_star:.3f}')
    return
