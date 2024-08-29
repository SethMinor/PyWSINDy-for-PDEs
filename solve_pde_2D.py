import scipy.fftpack as fft
from scipy.integrate import solve_ivp

def solve_pde_2D(phi,u,v, weights, x,y,t, tstart,tfinal, tmethod):

  # INTEGRATE SCALAR FIELD ADVECTED BY FLUID
  # Spectral in space, variable in time
  #------------------------------------------
  # phi = initial condition (2D numpy mesh)
  # u,v = full wind speeds (3D numpy meshes)
  # weights = PDE weights ([w1, ..., wn])
  # x,y,t = WSINDy grid (possibly distinct)
  # tstart = start time for simulation
  # tfinal = final time for simulation
  # tmethod = time integration method (scipy)
  #------------------------------------------

  # Real-space grid
  Lx = x[-1] - x[0]
  Ly = y[-1] - y[0]

  # Spatial resolution
  dx = x[1] - x[0]
  dy = y[1] - y[0]
  Nx = x.shape[0]
  Ny = y.shape[0]

  # Create spectral grid
  kx = np.fft.fftfreq(Nx, dx/(2*np.pi))
  ky = np.fft.fftfreq(Ny, dy/(2*np.pi))
  kx, ky = np.meshgrid(kx, ky, indexing='ij')

  # Dealiasing routine (Nx x Ny mesh)
  dealiase = np.abs(kx) < (2./3.)*(np.pi/dx)
  dealiase = dealiase & (np.abs(ky) < (2./3.)*(np.pi/dy))

  # Solution storage times
  dt = t[1] - t[0]
  n0 = find_nearest_index(t, tstart)
  nf = find_nearest_index(t, tfinal)
  t_eval = t[n0:nf]

  # SciPy options and parameters
  phi0 = phi.flatten().tolist()
  args = (u, v, t, weights, kx, ky, dealiase, Nx, Ny)
  scipy_params = {'method':tmethod, 'args':args, 't_eval':t_eval}

  # Time integration (SciPy)
  results = solve_ivp(RHS, [tstart, tfinal], phi0, **scipy_params)
  return results


# HELPER FUNCTIONS

# Find the index of an array element nearest to a target value
def find_nearest_index(array, target_value):
  return (np.abs(array - target_value)).argmin()

# Build discovered PDE from library of terms
def RHS(tsim, X, u, v, t, weights, kx, ky, dealiase, Nx, Ny):

  # Unpack state vector (list)
  phi = np.array(X).reshape((Nx, Ny))

  # Get closest indices
  n = find_nearest_index(t, tsim)
  un = np.copy(u[:,:,n])
  vn = np.copy(v[:,:,n])

  # Assemble PDE and listify (for SciPy)
  div = weighted_div(phi, un, vn, weights, kx, ky, dealiase)
  rhs = -div
  return rhs.flatten().tolist()


# TERM LIBRARY

# Weighted Cartesian divergence
# Div(phi*U) = w1*(phi*u)_x + w2*(phi*v)_y
def weighted_div(phi, un, vn, weights, kx, ky, dealiase):

  # Compute the products in physical space, then FFT
  phi_u = phi * un
  phi_v = phi * vn
  phi_u_hat = fft.fft2(phi_u)
  phi_v_hat = fft.fft2(phi_v)

  # Discovered weights
  (w1, w2) = (weights[0], weights[1])

  # Compute derivative in Fourier space
  div_hat = w1*1j*kx*phi_u_hat + w2*1j*ky*phi_v_hat
  div_hat *= dealiase

  # Transform solution back to real space and return
  div = np.real(fft.ifft2(div_hat))
  return div
