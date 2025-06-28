# WSINDy for PDEs
A Python 3 implementation of the Weak-form Sparse Identification of Nonlinear Dynamics (WSINDy) algorithm for partial differential equations.

- Based on the [JCP paper by **D. A. Messenger**, **D. M. Bortz** (2021)](https://www.sciencedirect.com/science/article/pii/S0021999121004204)
- See the original authors' [**MatLab** code repository](https://github.com/MathBioCU/WSINDy_PDE)
- Also see [this implemetation of **PyWSINDy for ODEs**](https://github.com/MathBioCU/PyWSINDy_ODE)

For other existing implementations, also see the [PySINDy documentation](https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html).
###### Stable as of June, 2025.
[![Python 3.11](https://img.shields.io/badge/python-%3E=3.11-blue?logo=python)](https://img.shields.io/badge/python-%3E=3.11-blue?logo=python)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/PyWSINDy-for-PDEs/blob/main/wsindy_examples.ipynb)
---
![wsindy_github_pic](https://github.com/SethMinor/WSINDy-for-Python/assets/97004318/8e567430-7368-420c-bf94-6eee224f7dc5)
## Notebooks
- ### `WSINDy.ipynb`<br>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/WSINDy-for-Python/blob/main/WSINDy.ipynb)<br><sub> A template for running WSINDy on your own data. See the 'Usage' section below for details. <br><sup> Stable as of August, 2024. </sup></sub>
- ### `WSINDy_Tutorial.ipynb`<br>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/WSINDy-for-Python/blob/main/WSINDy_Tutorial.ipynb)<br><sub> This notebook serves as a walkthrough and introduction to the WSINDy algorithm. As an example, it shows how the Kuramoto-Sivashinksy equation can be recovered from data (see the picture above). <br><sup> Stable as of July, 2024. </sup></sub>
- ### `WSINDy_SH23.ipynb`<br>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/WSINDy-for-Python/blob/main/WSINDy_SH23.ipynb)<br><sub> The WSINDy algorithm applied to the Swift-Hohenberg (23) equation. Simulation data were obtained using MatLab's *chebfun* package; see `sh23_simulation.m` and [chebfun.org](https://chebfun.org) (navigate to `examples > Swift Hohenberg`). <br><sup> Stable as of July, 2024. </sup></sub>
- ### `JHTDB_WSINDy.ipynb`<br>[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/WSINDy-for-Python/blob/main/JHTDB_WSINDy.ipynb)<br><sub> Uses a numerical simulation of the ideal MHD equations, sourced from the [Johns Hopkins Tubulence Database](https://turbulence.pha.jhu.edu/Forced_MHD_turbulence.aspx), as a dataset for WSINDy. <br><sup> Stable as of August, 2024. </sup></sub>
###### To access a dataset stored in Google Drive (e.g., `/content/drive/My Drive/WSINDy/dataset_name.txt`) while using Google Colab, use the following commands to change directories:
```python
# Create directory if necessary
!mkdir -p "/content/drive/My Drive/WSINDy"

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/My Drive/WSINDy
```
## Python Files
- ### `wsindy.py` <br> [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/) <br><sub> Returns sparse weights for candidate basis functions. See the 'Usage' section below for details. <br><sup> Stable as of July, 2024. </sup></sub>
## Dependencies
###### This codebases uses the following modules and naming conventions:
```python3
import torch
import scipy
import numpy as np
import itertools
import symengine as sp

import torch.linalg as la
from scipy.signal import convolve
from scipy.special import factorial
import matplotlib.pyplot as plt
from tqdm import tqdm

from wsindy import *
from helper_fcns import *
```
## Usage
###### For a dataset `U` (tensor), function library `fj` (dictionary), and derivative library `alpha` (tuple), the syntax is as follows:
```python
w = wsindy(U, fj, alpha, **params)
```
###### Example algorithm hyperparameter specification:
```python
# Grid parameters (should match dimension of dataset)
(Lx, Ly, T) = (30*np.pi, 30*np.pi, 20)
(dx, dy, dt) = (Lx/U.shape[0], Ly/U.shape[1], T/U.shape[-1])

# Function library
fields = 1 # Number of scalar fields
powers = 4 # Maximum monomial power
poly = get_poly(powers, fields)
trig = () # (Frequency, phase) pairs
fj = {'poly': poly, 'trig': trig}

# Derivative library
lhs = ((0,0,1),) # Evolution operator D^0
dimension = 2 # Spatial dimensions
pure_derivs = 4 # Include up to fourth order
cross_derivs = 2 # Include up to second order
rhs = get_alpha(dimension, pure_derivs, cross_derivs)
alpha = lhs + rhs

params = {
    # x = spatial domain(s)
    # dx = spatial discretization(s)
    # t = temporal domain
    # dt = temporal discretization
    # aux_fields = extra library variables
    # aux_scales = scaling factors for aux fields
    #--------------------------------------------
    'x' : [(0, Lx), (0, Ly)],
    'dx' : [dx, dy],
    't' : (0, T),
    'dt' : dt,

    # m = explicit (mx,...,mt) values
    # s = explicit (sx,...,st) values
    # lambdas = MSTLS threshold search space
    # threshold = known optimal threshold
    # p = explicit (px,...,pt) values
    # tau = test function tolerance
    # tau_hat = Fourier test function tolerance
    # scales = explicit (yu,yx,yt) scaling factors
    # M = explicit scaling matrix
    #---------------------------------------------

    # verbosity = report info and create plots? (0 or 1)
    # init_guess = [x0, y0, m1, m2], for (kx,kt) curve fit
    # max_its = specify maximum number of MSTLS iterations
    # sigma_NR = noise ratio of artifical gaussian noise
    # sparsify = use 'original' or 'scaled' data in MSTLS
    #-----------------------------------------------------
    'verbosity' : 1,
    'sigma_NR' : 0.0,
    'sparsify' : 'original'}
```

### Library Creation
The user defines the candidate terms in the library.

#### Default Library
To create a library of homogeneous monomials and their derivatives, use:
```python
[G,powers,derivs,rhs_names] = model.create_default_library()
model.set_library(G, powers, derivs, rhs_names)
```

#### Custom Library
Specify a sequence of terms $g(i,j)$ by providing lists of $\big(\mathbf{g}, \, j, \, i, \,$ `name` $\big)$ information:
```python
G = [tensor1, ..., tensorN]
powers = [int1, ..., intN]
derivs = [int1, ..., intN]
rhs_names = [string1, ..., stringN]

model.set_library(G, powers, derivs, rhs_names)
```
For fully non-homogeneous functions, set $j$ such that $\beta_j =$ `powers[j] = [0, ..., 0]`. <br> Also see the following helper functions:
- `model.compute_weak_poly(...)`
- `model.compute_weak_multipoly(...)`
- `model.compute_weak_trig(...)`
Convolved tensors should be evaluated over query points $\{(\boldsymbol{x}_k, t_k)\}$ using `tensor[model.conv_mask]`. For example:
```python
g = compute_weak_poly(model.U, kernels, model.spacing, yu=model.yu, yxyt=yxyt)
g = g[model.conv_mask]
```
