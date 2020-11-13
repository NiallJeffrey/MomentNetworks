# MomentNetworks
Demonstration of MomentNetworks for high-dimensional probability density estimation (LFI)

NOTE - this code is under rearrangement

## Test A
### 20 parameters, 20 data elements, and 1000000 training data realisations

#### (I)
Estimate the full (20,20) dimensional distribution P(d | \theta )) with MAF and compare with MCMC from the known likelihood. MAF fails completely in this set-up. If the number parameters is reduced to 12, DELFI does not fail:

[/notebooks_testA/delfi_failure_example.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/notebooks_testA/delfi_failure_example.ipynb)

#### (II)
Estimate the variance of the marginal distributions P(d|\theta_i)) and compare with MCMC from the known likelihood. The Moment Network perform well --  approx. 4 per cent accuracy for the estimated variance: 

[/notebooks_testA/delfi_moment_network.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/notebooks_testA/delfi_moment_network.ipynb)

#### (III)
Estimate the the marginal distributions P(d|\theta_i)) with MAF and compare with MCMC from the known likelihood. 

[/notebooks_testA/marginal_delfi_example.ipynb](https://github.com/NiallJeffrey/MomentNetworks/blob/master/notebooks_testA/marginal_delfi_example.ipynb)

