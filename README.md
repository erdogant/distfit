# distfit - Probability density fitting

[![Python](https://img.shields.io/pypi/pyversions/distfit)](https://img.shields.io/pypi/pyversions/distfit)
[![PyPI Version](https://img.shields.io/pypi/v/distfit)](https://pypi.org/project/distfit/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/distfit/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/distfit/month)](https://pepy.tech/project/distfit/month)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/distfit/)

### Background
    Python package for probability density fitting across 89 univariate distributions to non-censored data by residual sum of squares (RSS), and hypothesis testing.
    Probability density fitting is the fitting of a probability distribution to a series of data concerning the repeated measurement of a variable phenomenon.
    ``distfit`` scores each of the 89 different distributions for the fit wih the emperical distribution and return the best scoring distribution.

<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/example.png" width="650" />
</p>

### Functionalities

The ``distfit`` library is created with classes to ensure simplicity in usage.

```python
# Import library
from distfit import distfit

dist = distfit()        # Specify desired parameters
dist.fit_transform(X)   # Fit distributions on emperical data X
dist.predict(y)         # Predict the probability of the resonse variables
dist.plot()             # Plot the best fitted distribution (y is included if prediction is made)
```

### Contents
- [Installation](#-installation)
- [Examples](#-Examples)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
    Install distfit from PyPI (recommended). distfit is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 

#### Install from PyPi
```
pip install distfit
```

#### Install directly from github source (beta version)
```bash
pip install git+https://github.com/erdogant/distfit#egg=master
```  

#### Install by cloning  (beta version)
```bash
git clone https://github.com/erdogant/distfit.git
cd distfit
pip install -U .
```  


#### Check version number
```python
import distfit
print(distfit.__version__)
```

### Examples

Import ``distfit`` library

```python
from distfit import distfit
```

Create Some random data and model using default parameters:

```python
import numpy as np
X = np.random.normal(0, 2, [100,10])
y = [-8,-6,0,1,2,3,4,5,6]
```

Specify ``distfit`` parameters. In this example nothing is specied and that means that all parameters are set to default.
```python
dist = distfit()
dist.fit_transform(X)
dist.plot()

# Prints the screen:
# [distfit] >fit..
# [distfit] >transform..
# [distfit] >[norm      ] [RSS: 0.0133619] [loc=-0.059 scale=2.031] 
# [distfit] >[expon     ] [RSS: 0.3911576] [loc=-6.213 scale=6.154] 
# [distfit] >[pareto    ] [RSS: 0.6755185] [loc=-7.965 scale=1.752] 
# [distfit] >[dweibull  ] [RSS: 0.0183543] [loc=-0.053 scale=1.726] 
# [distfit] >[t         ] [RSS: 0.0133619] [loc=-0.059 scale=2.031] 
# [distfit] >[genextreme] [RSS: 0.0115116] [loc=-0.830 scale=1.964] 
# [distfit] >[gamma     ] [RSS: 0.0111372] [loc=-19.843 scale=0.209] 
# [distfit] >[lognorm   ] [RSS: 0.0111236] [loc=-29.689 scale=29.561] 
# [distfit] >[beta      ] [RSS: 0.0113012] [loc=-12.340 scale=41.781] 
# [distfit] >[uniform   ] [RSS: 0.2481737] [loc=-6.213 scale=12.281] 
```

<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1.png" width="450" />
</p>

Note that the best fit should be [normal], as this was also the input data. 
However, many other distributions can be very similar with specific loc/scale parameters. 
It is however not unusual to see gamma and beta distribution as these are the "barba-pappas" among the distributions. 
Lets print the summary of detected distributions with the Residual Sum of Squares.

```python
# All scores of the tested distributions
print(dist.summary)

# Distribution parameters for best fit
dist.model

# Make plot
dist.plot_summary()
```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1_summary.png" width="450" />
</p>

After we have a fitted model, we can make some predictions using the theoretical distributions. 
After making some predictions, we can plot again but now the predictions are automatically included.

```python
dist.predict(y)
dist.plot()
# 
# Prints to screen:
# [distfit] >predict..
# [distfit] >Multiple test correction..[fdr_bh]
```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1_prediction.png" width="450" />
</p>

The results of the prediction are stored in ``y_proba`` and ``y_pred``
```python

# Show the predictions for y
print(dist.y_pred)
# ['down' 'down' 'none' 'none' 'none' 'none' 'up' 'up' 'up']

# Show the probabilities for y that belong with the predictions
print(dist.y_proba)
# [2.75338375e-05 2.74664877e-03 4.74739680e-01 3.28636879e-01 1.99195071e-01 1.06316132e-01 5.05914722e-02 2.18922761e-02 8.89349927e-03]
 
# All predicted information is also stored in a structured dataframe
print(dist.df)
#    y   y_proba y_pred         P
# 0 -8  0.000028   down  0.000003
# 1 -6  0.002747   down  0.000610
# 2  0  0.474740   none  0.474740
# 3  1  0.328637   none  0.292122
# 4  2  0.199195   none  0.154929
# 5  3  0.106316   none  0.070877
# 6  4  0.050591     up  0.028106
# 7  5  0.021892     up  0.009730
# 8  6  0.008893     up  0.002964
```

Example if you want to test one specific distribution, such as the normal distribution:

```python
dist = distfit(distr='norm')
dist.fit_transform(X)

# [distfit] >fit..
# [distfit] >transform..
# [distfit] >[norm] [RSS: 0.0151267] [loc=0.103 scale=2.028]

dist.plot()
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

### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
	Contributions are welcome.
	See [LICENSE](LICENSE) for details.
