# distfit

[![Python](https://img.shields.io/pypi/pyversions/distfit)](https://img.shields.io/pypi/pyversions/distfit)
[![PyPI Version](https://img.shields.io/pypi/v/distfit)](https://pypi.org/project/distfit/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/distfit/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/distfit)](https://pepy.tech/project/distfit)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* Python package for probability density fitting and hypothesis testing.
* Probability density fitting is the fitting of a probability distribution to a series of data concerning the repeated measurement of a variable phenomenon. distfit scores each of the 89 different distributions for the fit wih the emperical distribution and return the best scoring distribution.

## The following functions are available:
```python
# To make the distribution fit with the input data
.fit()
# Compute probabilities using the fitted distribution
.proba_parametric()
# Compute probabilities in an emperical manner
.proba_emperical()
# Plot results
.plot()

See below for the exact working of the functions.
```

## Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install distfit from PyPI (recommended). distfit is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

## Requirements
```python
pip install numpy pandas matplotlib
```

## Quick Start
```
pip install distfit
```

* Alternatively, install distfit from the GitHub source:
```bash
git clone https://github.com/erdogant/distfit.git
cd distfit
python setup.py install
```  
#### Import distfit package
```python
import distfit as dist
```
#### Generate some random data:
```python
import numpy as np
X=np.random.normal(5, 8, [1000])

# Print to screen
print(X)
array([[-12.65284521,  -3.81514715,  -4.53613236],
       [ 11.5865475 ,   2.42547023,   6.6395518 ],
       [  3.82076163,   6.65765319,   9.95795751],
       ...,
       [  3.65728268,   7.298237  ,  -4.25641318],
       [  7.51820943,  16.26147929,  -0.60033084],
       [  2.49165326,   3.97880574,   7.98986818]])
```
#### Example fitting best scoring distribution to input-data:
```python
model = dist.fit(X)
dist.plot(model)
```
#### Output looks like this:
```
[DISTFIT] Checking for [norm] [SSE:0.000152]
[DISTFIT] Checking for [expon] [SSE:0.021767] 
[DISTFIT] Checking for [pareto] [SSE:0.054325] 
[DISTFIT] Checking for [dweibull] [SSE:0.000721]
[DISTFIT] Checking for [t] [SSE:0.000139]
[DISTFIT] Checking for [genextreme] [SSE:0.050649]
[DISTFIT] Checking for [gamma] [SSE:0.000152]
[DISTFIT] Checking for [lognorm] [SSE:0.000156]
[DISTFIT] Checking for [beta] [SSE:0.000152]
[DISTFIT] Checking for [uniform] [SSE:0.015671] 
[DISTFIT] Estimated distribution: t [loc:5.239912, scale:7.871518]

note that the best fit should be [normal], as this was also the input data. 
However, many other distributions can be very similar with specific loc/scale parameters. 
In this case, the t-distribution scored slightly better then normal. The normal distribution 
scored similar to gamma and beta which is not strange to see. 
```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1.png" width="400" />
</p>

#### Example Compute probability whether values are of interest compared 95%CII of the data distribution:
```python
expdata=[-20,-12,-8,0,1,2,3,5,10,20,30,35]
# Use fitted model
model_P = dist.proba_parametric(expdata, X, model=model)
# Make plot
dist.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig2a.png" width="200" />
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig2b.png" width="400" />
</p>

```python
# Its also possible to do the distribution fit in the proba_ function. Note that this if not practical in a loop with fixed background. 
model_P = dist.proba_parametric(expdata, X)
```


### Citation
Please cite distfit in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019distfit,
  title={distfit},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/distfit}},
}
```

### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* Contributions are welcome.

### Licence
See [LICENSE](LICENSE) for details.

### Donation
* This work is created and maintained in my free time. If this package is usefull to you and if want to see more like this, you can show your <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">gratitude</a> :) Thanks!
