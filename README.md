# distfit

[![Python](https://img.shields.io/pypi/pyversions/distfit)](https://img.shields.io/pypi/pyversions/distfit)
[![PyPI Version](https://img.shields.io/pypi/v/distfit)](https://pypi.org/project/distfit/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/distfit/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/distfit)](https://pepy.tech/project/distfit)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* Python package for probability density fitting and hypothesis testing.
* Probability density fitting is the fitting of a probability distribution to a series of data concerning the repeated measurement of a variable phenomenon. 
* distfit scores each of the 89 different distributions for the fit wih the emperical distribution and return the best scoring distribution.

### The following functions are available:
```python
import distfit as dist
# To make the distribution fit with the input data
dist.fit()
# Compute probabilities using the fitted distribution
dist.proba_parametric()
# Compute probabilities in an emperical manner
dist.proba_emperical()
# Plot results
dist.plot()
# Plot summary
dist.plot_summary()

See below for the exact working of the functions.
```

### Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install distfit from PyPI (recommended). distfit is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

### Requirements
```python
pip install numpy pandas matplotlib
```
### Quick Start
```
pip install distfit
```
### Alternatively, install distfit from the GitHub source:
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
X = np.random.beta(5, 8, [100,100])
# or 
# X = np.random.beta(5, 8, 1000)
# or anything else

# Print to screen
print(X)
# array([[-12.65284521,  -3.81514715,  -4.53613236],
#        [ 11.5865475 ,   2.42547023,   6.6395518 ],
#        [  3.82076163,   6.65765319,   9.95795751],
#        ...,
#        [  3.65728268,   7.298237  ,  -4.25641318],
#        [  7.51820943,  16.26147929,  -0.60033084],
#        [  2.49165326,   3.97880574,   7.98986818]])
```

#### Example fitting best scoring distribution to input-data:
```python
model = dist.fit(X)
dist.plot(model)

# Output looks like this:
# [DISTFIT.fit] Fitting [norm      ] [SSE: 1.1641360] [loc=0.384 scale=0.128] 
# [DISTFIT.fit] Fitting [expon     ] [SSE: 82.9253587] [loc=0.037 scale=0.347] 
# [DISTFIT.fit] Fitting [pareto    ] [SSE: 100.6452574] [loc=-0.711 scale=0.749] 
# [DISTFIT.fit] Fitting [dweibull  ] [SSE: 3.0304725] [loc=0.376 scale=0.112] 
# [DISTFIT.fit] Fitting [t         ] [SSE: 1.1640207] [loc=0.384 scale=0.128] 
# [DISTFIT.fit] Fitting [genextreme] [SSE: 0.4763435] [loc=0.335 scale=0.123] 
# [DISTFIT.fit] Fitting [gamma     ] [SSE: 0.6668446] [loc=-0.514 scale=0.018] 
# [DISTFIT.fit] Fitting [lognorm   ] [SSE: 0.6960495] [loc=-1.046 scale=1.424] 
# [DISTFIT.fit] Fitting [beta      ] [SSE: 0.3419988] [loc=-0.009 scale=0.987] 
# [DISTFIT.fit] Fitting [uniform   ] [SSE: 56.8836516] [loc=0.037 scale=0.797] 

```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1.png" width="400" />
</p>

Note that the best fit should be [beta], as this was also the input data. 
However, many other distributions can be very similar with specific loc/scale parameters. 
In this case, the beta-distribution scored best. 
It is however not unusual to see gamma and beta distribution as these are the "barba-pappas" among the distributions. 

* Summary of the SSE scores:
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig3.png" width="400" />
</p>

#### Example Compute probability whether values are of interest compared 95%CII of the data distribution:
This can be done using a pre-trained model or in simply in one run.
```python
X = np.random.beta(5, 8, [100,100])
y = [-1,-0.8,-0.6,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.5]

# Fit model (manner 1)
model = dist.fit(X)
out = dist.proba_parametric(y, model=model)

# Fit model and predict (manner 2) 
# Note that this if not practical in a loop with fixed background
out = dist.proba_parametric(y, X)

# print probabilities
print(out['proba'])

#   data             P          Padj bound
#   -1.0  0.000000e+00  0.000000e+00  down
#   -0.8  0.000000e+00  0.000000e+00  down
#   -0.6  0.000000e+00  0.000000e+00  down
#    0.0  1.559231e-08  3.563956e-08  down
#    0.1  4.467564e-03  7.148102e-03  down
#    0.2  7.085374e-02  8.720461e-02  none
#    0.3  2.726085e-01  2.907824e-01  none
#    0.4  4.390847e-01  4.390847e-01  none
#    0.5  1.905598e-01  2.177826e-01  none
#    0.6  5.360688e-02  7.147584e-02  none
#    0.7  7.935965e-03  1.154322e-02    up
#    0.8  3.697836e-04  6.573931e-04    up
#    0.9  8.037999e-07  1.607600e-06    up
#    1.0  0.000000e+00  0.000000e+00    up
#    1.1  0.000000e+00  0.000000e+00    up
#    1.5  0.000000e+00  0.000000e+00    up

# Make plot
dist.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig2b.png" width="400" />
</p>


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
* This work is created and maintained in my free time. If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated.
