# GRIB to CSV
# In a Google Colab notebook

# Get conda installer
!pip install -q condacolab
import condacolab
condacolab.install()
!conda --version
!conda install -c conda-forge eccodes

# Install ECMWF libraries
%pip install eccodes
%pip install cfgrib
!python -m eccodes selfcheck

# Find the grib file
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My Drive/WSINDy/ERA5

# Conversion to csv
import eccodes
import cfgrib
import pandas as pd

with cfgrib.open_dataset('era5_data.grib') as ds:
  df = ds.to_dataframe()
df['u'].to_csv('era5_u.txt', index=False, header=False, sep = ',')
df['v'].to_csv('era5_v.txt', index=False, header=False, sep = ',')
df['vo'].to_csv('era5_w.txt', index=False, header=False, sep = ',')
