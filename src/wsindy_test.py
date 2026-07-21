# WSINDy validation tests on known example systems
# Usage in terminal: "python wsindy_test.py"

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from scipy.integrate import solve_ivp
from wsindy import *

FAILURES = []

def check(label, passed, detail=''):
  print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f' ({detail})' if detail else ''))
  if not passed:
    FAILURES.append(label)

def simulate(rhs, ic, T, N):
  t_np = np.linspace(0, T, N)
  sol = solve_ivp(rhs, (0, T), ic, t_eval=t_np, rtol=1e-12, atol=1e-12, method='DOP853')
  return torch.tensor(t_np), [torch.tensor(ui) for ui in sol.y]

def fit_once(spec, U, V, names, lhs_name, m, Lambda, rescale=True):
  model = WSINDy(U, spec['alpha'], spec['beta'], spec['X'], V=V, names=names, m=m,
                 s=U.dim()*[1], verbosity=False, rescale=rescale, eqn_type=spec['equation_type'])
  [G, powers, derivs, rhs_names] = model.create_default_library()
  model.build_lhs(lhs_name)
  model.set_library(G, powers, derivs, rhs_names)
  w = model.MSTLS(Lambda=Lambda)
  got = {name: c for name,c in zip(rhs_names, w.tolist()) if c != 0}
  return model, got

def compare(got, truth):
  if set(got) != set(truth):
    return False, np.inf
  return True, max(abs(got[k]-v)/abs(v) for k,v in truth.items())

def noisy_cases(spec, noise, seed):
  torch.manual_seed(seed)
  states = [ui if noise == 0 else add_noise(ui, noise) for ui in spec['clean']]
  return spec['cases'](states)

def ode_cases(all_names, truths):
  def build(states):
    cases = []
    for i,truth in enumerate(truths):
      aux = [j for j in range(len(states)) if j != i]
      names = [all_names[i]] + [all_names[j] for j in aux]
      cases.append((states[i], [states[j] for j in aux], names, names[0]+'_{t}', truth))
    return cases
  return build

def test_system(spec):
  print(f"\n=== {spec['name']} ===")
  [m0, Lambda0] = [spec['m'], spec['Lambda']]

  # Coefficient support recovery and coefficient estimation on clean data
  for (U, V, names, lhs_name, truth) in noisy_cases(spec, 0, 0):
    [_, got] = fit_once(spec, U, V, names, lhs_name, m0, Lambda0)
    [ok, err] = compare(got, truth)
    check(f'{lhs_name} clean support', ok, ', '.join(got) if ok else f'got {got}')
    if ok:
      check(f'{lhs_name} clean coeffs', err < spec['tol_clean'], f'max rel err {err:.2e}')

  # Scaling with small additive i.i.d. Gaussian noise
  for noise in spec['noises']:
    [n_ok, n_total, errs] = [0, 0, []]
    for seed in range(spec['n_seeds']):
      for (U, V, names, lhs_name, truth) in noisy_cases(spec, noise, seed):
        [_, got] = fit_once(spec, U, V, names, lhs_name, m0, Lambda0)
        [ok, err] = compare(got, truth)
        n_total += 1
        if ok:
          n_ok += 1
          errs.append(err)
    max_err = max(errs) if errs else np.inf
    tol = spec['tol_noise'](noise)
    check(f'noise={noise:.4f} support {n_ok}/{n_total}, coeff err < {tol:.3f}',
          (n_ok == n_total) and (max_err < tol), f'max rel err {max_err:.2e}')

  # Hyperparameter sensitivity (m)
  m_values = [[max(2,int(round(f*mi))) for mi in m0] for f in spec['m_factors']]
  for (U, V, names, lhs_name, truth) in noisy_cases(spec, spec['sens_noise'], 0):
    base = WSINDy(U, spec['alpha'], spec['beta'], spec['X'], V=V, names=names, m=m0,
                  s=U.dim()*[1], verbosity=False, eqn_type=spec['eqn_type'])
    results = base.hyperparameter_sweep(lhs_name=lhs_name, m_values=m_values,
                                        Lambdas=[Lambda0], rescales=(True, False), plot=False)
    for i in range(len(results['Lambda'])):
      got = dict(zip(results['terms'][i], results['coeffs'][i]))
      [ok, err] = compare(got, truth)
      label = f"{lhs_name} m={results['m'][i]} rescale={results['rescale'][i]}"
      check(label, ok and (err < spec['tol_sens']),
            f'max rel err {err:.2e}' if ok else f"got {results['terms'][i]}")
  return

# LOGISTIC EQUATION
t, states = simulate(lambda t, u: 1.5*u*(1 - u/2.0), [0.2], 5, 501)
logistic = dict(name='Logistic', clean=states, X=t, eqn_type='ode',
                alpha=[[1],[0]], beta=[[0],[1],[2]], m=[16], Lambda=1e-1,
                cases=ode_cases(['u'], [{'(u)': 1.5, '(u^2)': -0.75}]),
                noises=[0.005, 0.01, 0.02], n_seeds=3, sens_noise=0.01,
                m_factors=[0.75, 1.0, 1.25],
                tol_clean=1e-8, tol_noise=lambda n: max(0.01, 2*n), tol_sens=0.05)

# LOTKA-VOLTERRA
t, states = simulate(lambda t, z: [1.5*z[0] - z[0]*z[1], -3.0*z[1] + z[0]*z[1]], [1.5, 1.0], 8, 1001)
lotka_volterra = dict(name='Lotka-Volterra', clean=states, X=t, eqn_type='ode',
                      alpha=[[1],[0]], beta=[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]],
                      m=[20], Lambda=1e-1,
                      cases=ode_cases(['x','y'], [{'(x)': 1.5, '(x y)': -1.0},
                                                  {'(y)': -3.0, '(y x)': 1.0}]),
                      noises=[0.005, 0.01, 0.02], n_seeds=3, sens_noise=0.01,
                      m_factors=[0.75, 1.0, 1.25],
                      tol_clean=1e-8, tol_noise=lambda n: max(0.01, 2*n), tol_sens=0.05)

# LORENZ '63
t, states = simulate(lambda t, s: [10.0*(s[1]-s[0]), s[0]*(28.0-s[2])-s[1], s[0]*s[1]-(8/3)*s[2]],
                     [-8.0, 8.0, 27.0], 3, 1501)
lorenz = dict(name="Lorenz '63", clean=states, X=t, eqn_type='ode',
              alpha=[[1],[0]],
              beta=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,0,0],
                    [1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]],
              m=[24], Lambda=1e-2,
              cases=ode_cases(['x','y','z'], [{'(x)': -10.0, '(y)': 10.0},
                                              {'(x)': 28.0, '(y)': -1.0, '(x z)': -1.0},
                                              {'(z)': -8/3, '(x y)': 1.0}]),
              noises=[0.0025, 0.005, 0.01], n_seeds=3, sens_noise=0.005,
              m_factors=[0.75, 1.0, 1.25],
              tol_clean=1e-8, tol_noise=lambda n: max(0.01, 2*n), tol_sens=0.05)

specs = [logistic, lotka_volterra, lorenz]

# KURAMOTO SIVASHINSKY
ks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Datasets', 'KS.txt')
if os.path.exists(ks_path):
  U0 = torch.tensor(np.loadtxt(ks_path, delimiter=','))
  x = torch.linspace(0, 32*np.pi, 256)
  t_ks = torch.linspace(0, 150, 301)
  ks = dict(name='Kuramoto-Sivashinsky', clean=[U0], X=[x, t_ks], eqn_type='pde',
            alpha=[[0,1],[0,0],[1,0],[2,0],[3,0],[4,0]], beta=[[0],[1],[2]],
            m=[18,30], Lambda=2.5e-2,
            cases=lambda states: [(states[0].view(256,-1), [], ['u'], 'u_{t}',
                                   {'(u^2)_{x}': -0.5, '(u)_{xx}': -1.0, '(u)_{xxxx}': -1.0})],
            noises=[0.01, 0.05], n_seeds=2, sens_noise=0.05,
            m_factors=[0.75, 1.0, 1.25],
            tol_clean=3e-2, tol_noise=lambda n: max(0.03, 2*n), tol_sens=0.05)
  specs.append(ks)
else:
  print(f'WARNING: {ks_path} not found, skipping Kuramoto-Sivashinsky tests.')

for spec in specs:
  test_system(spec)

print(f"\n{'ALL TESTS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S):'}")
for label in FAILURES:
  print(f'  {label}')
sys.exit(1 if FAILURES else 0)
