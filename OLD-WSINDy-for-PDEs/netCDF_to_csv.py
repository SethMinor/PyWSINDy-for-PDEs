# Access your Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change directories
%cd /content/drive/My Drive/WSINDy/ERA5/Global

%pip install netCDF4
import netCDF4
import pandas as pd
import numpy as np

nc = netCDF4.Dataset('era5_data.nc', mode='r')
print(nc.variables.keys())
print(nc.variables['pv'][:].shape)

pv = np.reshape(nc.variables['pv'][:], (96, 721*1440))
u = np.reshape(nc.variables['u'][:], (96, 721*1440))
v = np.reshape(nc.variables['v'][:], (96, 721*1440))
w = np.reshape(nc.variables['w'][:], (96, 721*1440))

np.savetxt('pv.csv', pv, delimiter=',')
np.savetxt('u.csv', u, delimiter=',')
np.savetxt('v.csv', v, delimiter=',')
np.savetxt('w.csv', w, delimiter=',')
