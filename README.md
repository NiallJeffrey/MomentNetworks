# MomentNetworks [![arXiv](https://img.shields.io/badge/arXiv-2011.05991-b31b1b.svg)](https://arxiv.org/abs/2011.05991) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Solving high-dimensional parameter inference: marginal posterior densities & Moment Networks 

[ArXiv paper (Accepted in the Third Workshop on Machine Learning and the Physical Sciences, NeurIPS 2020)](https://arxiv.org/abs/2011.05991)

## 100-dimensional model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NiallJeffrey/MomentNetworks/blob/master/MomentNetwork_demo/moment_network_100D.ipynb)

From a 100-dimensional parameter space, directly estimate the mean and covariance for pairs of parameters with Moment Networks:

[MomentNetwork_demo/moment_network_100D.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/MomentNetwork_demo/moment_network_100D.ipynb)

Google Colab notebook can be run in browser with GPU acceleration

(modules loaded with `!pip install 'git+https://github.com/NiallJeffrey/MomentNetworks.git'`)

## Gravitational wave data example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NiallJeffrey/MomentNetworks/blob/master/MomentNetwork_demo/MomentNetwork_grav_wave_demo.ipynb)

Demonstrate marginal posterior moment estimation using high-dimensional LIGO-like gravitational wave time series

[MomentNetwork_demo/MomentNetwork_grav_wave_demo.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/MomentNetwork_demo/MomentNetwork_grav_wave_demo.ipynb)


## 
### Dimensional challenges demos
#### 20 parameters, 20 data elements, and 1000000 training data realisations

#### (I)
Estimate the full (20,20) dimensional distribution P(d | \theta )) with MAF and compare with MCMC from the known likelihood. MAF fails completely in this set-up. If the number parameters is reduced to 12, MAF does not fail:

[MomentNetwork_demo/dimensional_challenges_demo/delfi_failure_example.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/MomentNetwork_demo/dimensional_challenges_demo/delfi_failure_example.ipynb)

#### (II)
Estimate the variance of the marginal distributions P(d|\theta_i)) and compare with MCMC from the known likelihood. The Moment Network perform well --  approx. 4 per cent accuracy for the estimated variance: 

[/MomentNetwork_demo/dimensional_challenges_demo/delfi_moment_network.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/MomentNetwork_demo/dimensional_challenges_demo/delfi_moment_network.ipynb)

#### (III)
Estimate the the marginal distributions P(d|\theta_i)) with MAF and compare with MCMC from the known likelihood. 

[/MomentNetwork_demo/dimensional_challenges_demo/marginal_delfi_example.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/MomentNetwork_demo/dimensional_challenges_demo/marginal_delfi_example.ipynb)

