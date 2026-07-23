from helper_fcns import *
import csv

class WSINDy:
  def __init__(self, U, alpha, beta, X, V=[], names=None, m=None, p=None, s=None, jacobian = 1.,
               tau=1e-10, tau_hat=2, init_kc_guess=[10,1,10,0], verbosity=True, rescale=True, eqn_type='pde'):
    self.U = U # state variable
    self.V = V # auxiliary variables
    self.alpha = alpha # derivative multi-indices
    self.beta = beta # monomial multi-indices
    self.X = X if type(X) == list else [X]
    self.eqn_type = eqn_type

    self.jacobian = jacobian # dx*...*dt
    self.tau = tau # test function support tolerance
    self.tau_hat = tau_hat # for spectral matching
    self.verbosity = verbosity
    self.init_kc_guess = init_kc_guess # for spectral matching
    self.rescale = rescale

    if eqn_type not in ('ode', 'pde'):
      raise ValueError("eqn_type must be 'ode' or 'pde'.")
    if eqn_type == 'ode':
      if self.U.dim() != 1:
        raise ValueError("ODE data must be one-dimensional in time.")
      if len(self.X) != 1:
        raise ValueError("ODE data requires a single time axis.")
    elif self.U.dim() < 2:
      raise ValueError("PDE data requires at least one spatial axis.")

    self.beta_max = max([bj[0] for bj in beta])
    self.dX = [Xi.diff()[0] for Xi in self.X]

    self.names = names if names is not None else ['u']+['v'+str(i+1) for i in range(len(V))]
    self.m = m if m is not None else self.compute_m() # test function support radii
    self.p = p if p is not None else self.compute_p() # test function degrees
    self.s = s if s is not None else [U.shape[i]//100 for i in range(U.ndim)] # subsampling rates

    # Scale-invariant preconditioning
    if rescale:
      [self.yx, self.yt] = self.compute_spatial_scales()
      self.yu = self.compute_u_scale(self.U, self.beta_max)
      self.aux_scales = []
      for i,Vi in enumerate(V):
        beta_max = max([bj[i+1] for bj in self.beta])
        self.aux_scales.append(self.compute_u_scale(Vi, beta_max))
    else:
      [self.yx, self.yt, self.yu, self.aux_scales] = 4*[None]

    [self.mask, self.flat_mask] = self.compute_query_points()
    [self.axes, self.kernels] = self.build_axes()
    self.test_fcns = self.build_test_fcns()
    self.derivative_names = self.get_derivative_names()
   
  def compute_m(self):
    m = [self.spectral_matching(d, changepoint, F_root) for d in range(self.U.dim())]
    return m

  def compute_p(self):
    p = [compute_degrees(d, md, self.alpha, tau=self.tau) for d,md in enumerate(self.m)]
    return p

  # Determines bandwidth of test functions by estimating signal-dominated modes
  def spectral_matching(self, d, changepoint, F_root):
    Nd = self.U.shape[d]
    Uhat_d = abs(torch.fft.rfft(self.U, n=Nd, dim=d))
    dims = [dim for dim in range(Uhat_d.ndimension()) if dim != d]
    Uhat_d = Uhat_d.mean(dim = dims) if dims else Uhat_d

    Hd = torch.cumsum(Uhat_d, dim=0) # Cumulative sum
    Hd = (Hd/Hd.max()).numpy() # Normalize for curve fitting

    # Solve change-point problem
    freqs = torch.arange(0, np.floor(Nd/2)+1, 1).numpy()
    params = scipy.optimize.curve_fit(changepoint, freqs, Hd, p0=self.init_kc_guess)[0]
    k = int(params[0])

    # Solve root-finding problem
    guess = (np.sqrt(3)*Nd*self.tau_hat)/(2*np.pi*k)
    guess = guess * (1 + np.sqrt(1 - (8/np.sqrt(3))*np.log(self.tau)))/2
    md = int(scipy.optimize.root(F_root, guess, args=(k,Nd,self.tau_hat,self.tau)).x[0])

    if self.verbosity:
      plt.figure(figsize=(7,2))
      Uhat_d = Uhat_d.numpy()
      label = r'$|\mathcal{F}[u]|(k)$' if self.eqn_type == 'ode' else r'$|\mathcal{F}_{x_d}[u]|(k)$'
      plt.plot(freqs, Uhat_d, '.-')
      plt.axvline(params[0], ls='--', color='r', label=r'Estimated changepoint, $\hat{k}_d$')
      plt.xlabel('Wavenumber, $k$')
      plt.ylabel(label)
      plt.title(fr'Spectral matching: $\hat{{m}}_d={md}$ (axis: $d=${d})')
      plt.legend(loc = 'upper right', framealpha=0.8, fontsize=11)
      plt.yscale('log')
      plt.grid(True, alpha=0.3, color='silver')
      plt.show()
    return md

  # Compute mask of query point indices, U[mask] = U[xk,...,tk]
  def compute_query_points(self):
    subsamples = [subsample(self.s[i], self.m[i], self.X[i]) for i in range(self.U.ndim)]
    cartesian_prod = itertools.product(*subsamples)
    mask = tuple(map(torch.tensor, zip(*cartesian_prod))) # for tensors
    flat_mask = tuple([mask[i]-self.m[i] for i in range(self.U.ndim)]) # for vectorized quantities

    if self.verbosity:
      if self.eqn_type == 'ode':
        t = self.X[0]
        tk = t[subsamples[0]]

        plt.figure(figsize=(7,2))
        plt.plot(t, self.U, '.-', zorder=1)
        plt.scatter(tk, self.U[mask], color='r', marker='.', s=14, label='Query points', zorder=2)
        plt.xlabel('$t$')
        plt.ylabel('$' + self.names[0] + '(t)$')
        plt.legend(loc='best', framealpha=0.8)
        plt.grid(True, alpha=0.3, color='silver')
        plt.show()
      else:
        [x1, x2] = [self.X[0], self.X[1]]
        [X1, X2] = np.meshgrid(self.X[0], self.X[1])
        [xk1, xk2] = [x1[subsamples[0]], x2[subsamples[1]]]
        [XK1, XK2] = [Xi.flatten() for Xi in np.meshgrid(xk1, xk2)]
        slice0 = (slice(None),slice(None)) + (self.U.dim()-2)*(0,)

        plt.figure(figsize=(6,6))
        plt.pcolormesh(X1, X2, self.U[slice0].T, cmap='coolwarm')
        plt.scatter(XK1, XK2, color='black', marker='.', s=2, label='Query Points')
        labels = ['x', 't'] if self.U.ndim==2 else ['x_1', 'x_2', ', \\dots']
        plt.xlabel('$' + labels[0] + '$')
        plt.ylabel('$' + labels[1] + '$')
        plt.title('$' + self.names[0] + '(' + labels[0] + ',' + labels[1:] + ')$')
        plt.legend(loc='upper right')
        plt.show()
    return mask, flat_mask

  # Compute scale for a state variable, yu
  def compute_u_scale(self, u, beta_max):
    U_2 = la.norm(u).item()
    U_beta = la.norm(u**beta_max).item()
    yu = (U_2 / U_beta)**(1 / max(1,beta_max))
    return yu

  # Compute spatio-temporal scales yx, yt (yx = [] for ODEs)
  def compute_spatial_scales(self):
    D = len(self.dX) - 1
    max_x = []
    for d in range(len(self.alpha[0]) - 1):
      max_d = max(tuple(ai[d] for ai in self.alpha))
      max_x.append(max(1,max_d))
    max_t = max(1,max(tuple(ai[-1] for ai in self.alpha)))

    # Ansatz given in the paper
    yx = [(1/(self.m[d]*self.dX[d]) * (my_nchoosek(self.p[d], max_x[d]/2)
          * factorial(max_x[d]))**(1/max_x[d])).item() for d in range(D)]
    yt = (1 / (self.m[-1] * self.dX[-1]) * (my_nchoosek(self.p[-1], max_t/2)
          * factorial(max_t))**(1/max_t)).item()
    return yx,yt

  # Compute scale matrix diagonal, M = diag(mu)
  def compute_scale_matrix(self, powers, derivs):
    D = self.U.dim() - 1
    [yx, yt] = [self.yx, self.yt]
    yu = [self.yu] + self.aux_scales

    num_terms = len(powers)
    mu = torch.zeros(num_terms, dtype=torch.float64)
    for j in range(num_terms):
      aj = derivs[j]
      bj = powers[j]
      if (aj==None) or (bj==None):
        mu[j] = 1.
      else:
        yx_exps = [yx[d]**(self.alpha[0][d] - self.alpha[aj][d]) for d in range(D)]
        yx_term = np.prod(yx_exps)
        yt_term = yt**(self.alpha[0][-1] - self.alpha[aj][-1])

        yu_term = [yu[n]**(bj[n]) for n in range(len(yu))]
        yu_term = np.prod(yu_term)

        mu[j] = (yu_term * yx_term * yt_term)/self.yu
    return mu

  # Returns symbolic derivatives
  def get_derivative_names(self):
    D = self.U.dim() - 1
    derivative_names = []
    for elem in self.alpha:
      if all(value == 0 for value in elem):
        derivative_names.append('')
      else:
        # (0+1)-D, (1+1)-D, (2+1)-D, (3+1)-D case-handling
        if D == 0:
          derivative_names.append('_{'+'t'*elem[0]+'}') # for ODEs
        elif D == 1:
          derivative_names.append('_{'+'t'*elem[1]+'x'*elem[0]+'}')
        elif D == 2:
          derivative_names.append('_{'+'t'*elem[2]+'x'*elem[0]+'y'*elem[1]+'}')
        elif D == 3:
          derivative_names.append('_{'+'t'*elem[3]+'x'*elem[0]+'y'*elem[1]+'z'*elem[2]+'}')
        else:
          raise ValueError("Whoah! Spatial dimension can only be: 0, 1, 2, or 3.")
    return derivative_names

  # Compute test function and its derivatives along d-th axis
  def get_weight_fcns(self, d):
    [x,m,p] = [self.X[d], self.m[d], self.p[d]]
    dx = (x[1] - x[0]).item()

    if (m > (len(x)-1)/2) or (m <= 1):
      raise ValueError('Error: invalid test function support.')

    if self.rescale:
      scale_d = (self.yx + [self.yt])[d]
      dx *= scale_d

    # Initialize grid of discretized test fcn values
    test_fcns_d = torch.zeros(len(self.alpha), 2*m+1, dtype=torch.float64)
    n_grid = torch.linspace(-1, 1, 2*m+1, dtype=torch.float64)
    multi_index_d = tuple(ai[d] for ai in self.alpha)

    # Compute symbolic derivatives
    x_sym = sp.Symbol('x')
    phi_bar = (1 - x_sym**2)**p
    vec = np.vectorize(D_phibar)

    for i in range(len(self.alpha)):
      if (i > 0) and (multi_index_d[i-1] == multi_index_d[i]):
        test_fcns_d[i,:] += test_fcns_d[i-1,:]
      else:
        num_derivs = multi_index_d[i]
        A_i = torch.from_numpy(vec(n_grid, num_derivs, x_sym, phi_bar))
        test_fcns_d[i,:] += (1/((m*dx)**num_derivs)) * A_i

    if self.verbosity:
      plt.figure(figsize=(7,2))
      for i in range(len(test_fcns_d[:,0])):
        plt.plot(m*dx*n_grid, test_fcns_d[i,:], '--.', label=f'$i={i}$')
      plt.title(f'Test function components (axis: $d={d}$)')
      if self.eqn_type == 'ode':
        plt.xlabel('$t_k - t$')
        plt.ylabel(r'$\left(\frac{d}{dt}\right)^{{\alpha_i}}\phi(t_k - t)$')
      else:
        plt.xlabel('$x_k - x$')
        plt.ylabel(fr'$D^{{\alpha_i}}\phi(x_k - x)$')
      plt.grid(True, alpha=0.3, color='silver')
      plt.legend(loc='upper right', framealpha=0.8)
      plt.show()
    return test_fcns_d

  # Build lists of vectorized test function components
  def build_axes(self):
    #axes = [self.get_weight_fcns(d) for d in range(len(self.m))]
    axes = [self.get_weight_fcns(d) for d in range(self.U.ndim)]
    kernels = [[axes[d][i,:] for d in range(self.U.ndim)] for i in range(len(self.alpha))]
    return axes, kernels

  # Build tensorized test functions (Nx x ... x Nt)
  def build_test_fcns(self):
    num_derivs = len(self.axes[0])
    D = len(self.axes)
    test_fcns = []
    for s in range(num_derivs):
      axes = [self.axes[d][s,:] for d in range(D)]
      ijk = [chr(105 + i) for i in range(D)]
      einsum = ','.join(ijk) + '->' + ''.join(ijk)
      D_phi = torch.einsum(einsum, *axes)
      test_fcns.append(D_phi)
    return test_fcns

  # Weak time derivative
  def build_lhs(self, lhs_name):
    kernel = self.kernels[0]
    if self.rescale:
      yxyt = np.prod(self.yx + [self.yt])
      lhs = compute_weak_poly(self.U, kernel, self.dX, yu=self.yu, yxyt=yxyt, jacobian=self.jacobian)
    else:
      lhs = compute_weak_poly(self.U, kernel, self.dX, jacobian=self.jacobian)
    b = lhs[self.flat_mask]
    self.lhs_name = lhs_name
    self.lhs = b
    return

  # Computes default monomial library terms
  def create_default_library(self):
    [G, powers, derivs, rhs_names] = [],[],[],[]
    state = [self.U] + self.V
    if self.rescale:
      yu = [self.yu] + self.aux_scales
      yxyt = np.prod(self.yx + [self.yt])
    else:
      yu = len(state) * [1.]
      yxyt = 1.

    for i,ai in enumerate(tqdm(self.alpha[1:]), start=1):
      kernel = self.kernels[i]
      for j,bj in enumerate(self.beta):
        if all(bjd==0 for bjd in bj) and any(aid!=0 for aid in ai):
          continue # Avoid derivatives of constant terms
        else:
          assert (len(bj)-1) == len(self.V), "Inconsistent number of powers!"
          derivs.append(i)
          powers.append(bj)
          term = compute_weak_multipoly(state, kernel, self.dX, power=bj, yu=yu, yxyt=yxyt, jacobian=self.jacobian)
          name = self.format_monomial(bj) + self.derivative_names[i]
          G.append(term[self.flat_mask])
          rhs_names.append(name)
    return G, powers, derivs, rhs_names

  # Fancy monomial formatting
  def format_monomial(self, bj):
    terms = []
    for d in range(len(bj)):
      if bj[d] == 0:
        continue
      if bj[d] == 1:
        terms.append(self.names[d])
      else:
        terms.append(f'{self.names[d]}^{bj[d]}')
    return '(' + ' '.join(terms) + ')' if terms else '(1)'

  def set_library(self, G, powers, derivs, rhs_names):
    G = torch.stack(G, dim=1)
    if G.shape[0] != len(self.mask[0]):
      raise ValueError("Library has inconsistent dimensions.")
    self.rhs_names = rhs_names
    self.library = G
    if self.rescale:
      self.mu = self.compute_scale_matrix(powers, derivs)
    return

  # Full MSTLS optimization routine, scans through Lambdas
  #def MSTLS(self, Lambda=None, Lambdas=10**((4/49)*torch.arange(0,50)-4)):
  def MSTLS(self, Lambda=None, Lambdas=10**((3/99)*torch.arange(0,100)-3)):
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

  # Modified sequential thresholding least squares (MSTLS) routine
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

  # Prints a report of the WSINDy run
  def print_report(self):
    print('\nHYPER-PARAMETERS')
    print(f'm = {self.m}')
    print(f'p = {self.p}')
    print(f's = {self.s}')
    if self.rescale:
      if self.eqn_type == 'ode':
        print(f'yt = {self.yt:.3f}')
      else:
        scales = self.yx + [self.yt]
        print('[yx, yt] = ' + str([float(f" {yi:.3f}") for yi in scales]))
      print(f'yu = {self.yu:.3f}')
      aux_scales = [float(f'{yu:.3f}') for yu in self.aux_scales]
      print(f'Aux. scales = {aux_scales}\n')
    else:
      print('Not rescaled.\n')

    print('LIBRARY')
    print(f'Number of query points = {self.library.shape[0]}')
    print(f'Number of candidate terms = {self.library.shape[1]}')
    print(f'Condition number = {la.cond(self.library):.2e}\n')

    print('RESULTS')
    eq_label = self.eqn_type.upper()
    eqn = symbolic_eqn(self.lhs_name, self.rhs_names, self.coeffs)
    num_terms = self.coeffs.count_nonzero().item()
    if self.rescale:
      [r, R2] = compute_residuals(self.library, self.coeffs/self.mu, self.lhs)
    else:
      [r, R2] = compute_residuals(self.library, self.coeffs, self.lhs)
    print(f'{eq_label}: {eqn}')
    print(f'Nonzero terms = {num_terms}')
    print(f'Relative L2 error = {la.norm(r)/la.norm(self.lhs):.3f}')
    print(f'R^2 = {R2:.3f}')
    print(f'Lambda = {self.Lambda:.2e}')
    print(f'Loss = {self.loss:.3f}')
    return

  # Sweep hyperparameters (m, Lambda) with rescale = True/False
  def hyperparameter_sweep(self, lhs_name=None, m_values=None, Lambdas=None, rescales=(True, False),
                           library_fcn=None, true_coeffs=None, csv_path=None, plot=True):
    if lhs_name is None:
      lhs_name = getattr(self, 'lhs_name', self.names[0] + self.derivative_names[0])
    if m_values is None:
      m_values = []
      #for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
      for factor in np.linspace(0.5, 1.5, 40):
        m_new = [int(round(factor*mi)) for mi in self.m]
        m_new = [min(max(mi,2), (self.U.shape[i]-1)//2) for i,mi in enumerate(m_new)]
        if m_new not in m_values:
          m_values.append(m_new)
    else:
      m_values = [(self.U.ndim*[mi] if type(mi)==int else list(mi)) for mi in m_values]
    if Lambdas is None:
      #Lambdas = 10**((4/49)*torch.arange(0,50)-4)
      Lambdas = 10**((4/199)*torch.arange(0,200)-4)
    Lambdas = [float(Li) for Li in Lambdas]

    keys = ['m','p','Lambda','rescale','model','terms','coeffs','sparsity','loss','R2','rel_L2','cond_G']
    if true_coeffs is not None:
      keys.append('coeff_err')
    results = {key: [] for key in keys}

    s = self.U.ndim * [1]
    for m in tqdm(m_values):
      for rescale in rescales:
        model = WSINDy(self.U, self.alpha, self.beta, self.X, V=self.V, names=self.names, m=m, s=s, jacobian=self.jacobian, tau=self.tau,
                       tau_hat=self.tau_hat, init_kc_guess=self.init_kc_guess, verbosity=False, rescale=rescale, eqn_type=self.eqn_type)
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

          results['m'].append(str(m))
          results['p'].append(str(model.p))
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

          # Max relative coefficient error
          if true_coeffs is not None:
            got = dict(zip(results['terms'][-1], results['coeffs'][-1]))
            err = max(abs(got.get(name, 0.) - cj)/abs(cj) for name,cj in true_coeffs.items())
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

  # Heatmaps of hyperparameter sweep results over (m, Lambda)
  def plot_sweep(self, results, metrics=None):
    if metrics is None:
      #metrics = ('sparsity', 'R2') + (('coeff_err',) if 'coeff_err' in results else ())
      metrics = ('sparsity', 'R2', 'loss') + (('coeff_err',) if 'coeff_err' in results else ())

    m_labels = list(dict.fromkeys(results['m']))
    Lambdas = list(dict.fromkeys(results['Lambda']))
    rescales = list(dict.fromkeys(results['rescale']))
    titles = {'sparsity': r'Nonzero terms, $\|\mathbf{w}\|_0$',
              'R2': r'Equation fit, $R^2$',
              'loss': r'WSINDyLoss, $\mathcal{L}(\mathbf{w})$',
              'coeff_err': r'Coeff. Error, $\log_{10}(E_{\infty})$'}

    for rescale in rescales:
      #fig, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 0.5*len(m_labels)+1.8))
      fig, ax = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4), sharey=True)
      ax = np.atleast_1d(ax)
      for j,metric in enumerate(metrics):
        Z = np.full((len(m_labels), len(Lambdas)), np.nan)

        for i in range(len(results['Lambda'])):
          if results['rescale'][i] == rescale:
            Z[m_labels.index(results['m'][i]), Lambdas.index(results['Lambda'][i])] = results[metric][i]

        if metric == 'R2':
          pcm = ax[j].pcolormesh(Lambdas, range(len(m_labels)), Z, shading='nearest', cmap='Reds_r', vmin=0, vmax=1)
        elif metric == 'coeff_err':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, range(len(m_labels)), Z, shading='nearest', cmap='Reds')
        elif metric == 'loss':
          pcm = ax[j].pcolormesh(Lambdas, range(len(m_labels)), Z, shading='nearest', cmap='Reds')
        else:
          pcm = ax[j].pcolormesh(Lambdas, range(len(m_labels)), Z, shading='nearest', cmap='Reds')

        ax[j].set_xscale('log')
        ax[j].set_yticks(range(len(m_labels)))
        ax[j].set_yticklabels(m_labels, fontsize=7)
        ax[j].set_xlabel(r'$\lambda$')
        ax[j].set_ylabel('$m$')
        ax[j].set_title(titles.get(metric, metric))
        fig.colorbar(pcm, ax=ax[j])
      fig.suptitle(f'rescale = {rescale}')
      fig.tight_layout()
      plt.show()
    return
