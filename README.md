# WSINDy for Python
##### A Python 3 implementation of the Weak-form Sparse Identification of Nonlinear Dynamics (WSINDy) algorithm.
##### Based on the [JCP paper by **D. A. Messenger**, **D. M. Bortz** 2021](https://www.sciencedirect.com/science/article/pii/S0021999121004204). <br> See authors' original [MatLab code repository](https://github.com/MathBioCU/WSINDy_PDE) (copyright 2020, all rights reserved by original authors).
##### See also [PySINDy documentation](https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples/example.html).
###### Python code by Seth Minor. <br> Stable as of June, 2024.
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SethMinor/WSINDy-for-Python/blob/main/Weak_SINDy_example.ipynb)
---
![wsindy_github_pic](https://github.com/SethMinor/WSINDy-for-Python/assets/97004318/8e567430-7368-420c-bf94-6eee224f7dc5)
## Files Included
`Weak_SINDy_example.ipynb` is a Jupyter notebook containing a step-by-step example of WSINDy identifying the $(1+1)$-dimensional Kuramoto-Sivashinksy equation from data.

###### To access a dataset stored in Google Drive while using Google Colab (e.g., in `/content/drive/My Drive/WSINDy/dataset_name.txt`), use the following commands to change directories.
```python
!mkdir -p "/content/drive/My Drive/WSINDy"

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/My Drive/WSINDy
```
