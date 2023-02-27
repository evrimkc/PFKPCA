# PFKPCA
A Python Implementation of the "Projection-free Kernel Principal Component Analysis for Denoising" Article by A.T. Bui, J.-K. Im and D.W. Apley et al. / Neurocomputing 357 (2019) 163–176


## About
A Python implementation of the projection-free KPCA denoising approach by Bui et al. which, in order to denoise an observation, performs a single line search along the gradient descent direction of the squared projection error instead of the usual projection and subsequent preimage approximation steps with the premise that this moves an observation towards the underlying manifold that represents the noiseless data in the most direct manner possible.

## Code
This code repository contains all of the data and code to carry out the same implementation and visualizations. This repository contains the following files:
- [`data8`](./data8) - contains the digit 8 data from MNIST dataset
- [`pfkpca.py`](./pfkpca.py) - main script, contains code for preparing the MNIST data, kernel matrices, centering functions, cost functions, gradient functions, all the steps for the PFKPCA algorithm, performance measures, visualizations
- [pfkpca_toy.ipynb](./pfkpca_toy.ipynb) - contains code for generating a synthetic quadratic dataset, performing hyperparameter tuning for PFKPCA algorithm, visualization of principle components, steepest descent directions, and denoising
- [pfkpca_mnist_data8.ipynb](./pfkpca_mnist_data8.ipynb) - contains code for reading images and adding noise to images, running PFKPCA algoritm, visualization of images (actual, noisy, denoised) and the denoising process  

### Installation
Install [MatPlotLib](https://matplotlib.org/), [Numpy](https://numpy.org/), [tqdm](https://github.com/tqdm/tqdm), [Pandas](https://pandas.pydata.org/), [Scipy](https://scipy.org/), [NumDiffTools](https://github.com/pbrod/numdifftools)
```
conda install matplotlib
conda install numpy
conda install tqdm
pip install pandas
pip install scipy
pip install numdifftools
```