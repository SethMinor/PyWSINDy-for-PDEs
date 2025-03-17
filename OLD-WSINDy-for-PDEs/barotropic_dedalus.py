# Dedalus simulations for Weak SINDy Models
# Based on examples from https://dedalus-project.readthedocs.io/

# Equivalent Barotropic Model
# Simulation data from PyQG.

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Ly = 2*np.pi, 2*np.pi # Periodic domain
Nx, Ny = 256, 256
w = -0.96 # WSINDy weights
dealias = 3/2
dtype = np.float64

# Coordinate bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
zeta = dist.Field(name='zeta', bases=(xbasis, ybasis)) # Vorticity
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis)) # Velocity
tau = dist.Field(name='tau') # Incompressibility condition

# Substitutions
#dx = lambda A: d3.Differentiate(A, xcoord) # Partial wrt x, e.g. "dx(u)"
ex, ey = coords.unit_vector_fields(dist)

# Problem statement
#problem = d3.IVP([zeta, u], namespace=locals()) # M.dt(X) + L.X = F(X, t)
problem = d3.IVP([zeta, u, tau], namespace=locals())
problem.add_equation("dt(zeta) = w*u@grad(zeta) + w*div(u)*zeta") # Divergence
#problem.add_equation("dt(zeta) = w*u@grad(zeta)") # Advection
problem.add_equation("div(skew(u)) + zeta = 0") # Coupling zeta and u
problem.add_equation("div(u) + tau = 0") # Incompressibility
problem.add_equation("integ(zeta) = 0") # Need to pick a gauge for vorticity


# Initial conditions
x, y = dist.local_grids(xbasis, ybasis)
n = 2*Lx
zeta['g'] = np.log(1 + np.cosh(n)**2/np.cosh(n*(x-0.2*Lx))**2) / (2*n)
u['g'][0] = 1/2 + 1/2 * (np.tanh((y-0.5)/0.1) - np.tanh((y+0.5)/0.1)) # Shear
# Add small vertical velocity perturbations localized to the shear layers
u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(y-0.5)**2/0.01)

# Define solver
stop_sim_time = 5
timestepper = d3.RK222
max_timestep = 1e-2
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
snapshots.add_task(zeta, name='Vorticity, zeta')
snapshots.add_task(u, name='Velocity, u')
#snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL condition for adaptive timesteps
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2,
              threshold=0.1, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e'
                        %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

# # When the simulation is finished, plot the results
# plt.figure(figsize=(6, 4))
# plt.pcolormesh(x.ravel(), np.array(t_list), np.array(u_list),
#                cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
# plt.xlim(0, Lx)
# plt.ylim(0, stop_sim_time)
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title(f'KdV-Burgers, (a,b)=({a},{b})')
# plt.tight_layout()
# plt.show()
# plt.savefig('kdv_burgers.pdf')
# plt.savefig('kdv_burgers.png', dpi=200)
