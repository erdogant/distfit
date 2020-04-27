# distfit

[![Python](https://img.shields.io/pypi/pyversions/distfit)](https://img.shields.io/pypi/pyversions/distfit)
[![PyPI Version](https://img.shields.io/pypi/v/distfit)](https://pypi.org/project/distfit/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/distfit/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/distfit)](https://pepy.tech/project/distfit)

* Python package for probability density fitting and hypothesis testing.
* Probability density fitting is the fitting of a probability distribution to a series of data concerning the repeated measurement of a variable phenomenon. 
* distfit scores each of the 89 different distributions for the fit wih the emperical distribution and return the best scoring distribution.

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
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install distfit from PyPI (recommended). distfit is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

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
pip install git+https://github.com/erdogant/distfit
git clone https://github.com/erdogant/distfit.git
cd distfit
pip install -U .
```  


#### Check version number
```python
import distfit as distfit
print(distfit.__version__)
```

### Examples
```python
from distfit import distfit
```

#### Example 1:

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
# [distfit] >[norm      ] [SSE: 0.0133619] [loc=-0.059 scale=2.031] 
# [distfit] >[expon     ] [SSE: 0.3911576] [loc=-6.213 scale=6.154] 
# [distfit] >[pareto    ] [SSE: 0.6755185] [loc=-7.965 scale=1.752] 
# [distfit] >[dweibull  ] [SSE: 0.0183543] [loc=-0.053 scale=1.726] 
# [distfit] >[t         ] [SSE: 0.0133619] [loc=-0.059 scale=2.031] 
# [distfit] >[genextreme] [SSE: 0.0115116] [loc=-0.830 scale=1.964] 
# [distfit] >[gamma     ] [SSE: 0.0111372] [loc=-19.843 scale=0.209] 
# [distfit] >[lognorm   ] [SSE: 0.0111236] [loc=-29.689 scale=29.561] 
# [distfit] >[beta      ] [SSE: 0.0113012] [loc=-12.340 scale=41.781] 
# [distfit] >[uniform   ] [SSE: 0.2481737] [loc=-6.213 scale=12.281] 
```

<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1.png" width="600" />
</p>

Note that the best fit should be [normal], as this was also the input data. 
However, many other distributions can be very similar with specific loc/scale parameters. 
It is however not unusual to see gamma and beta distribution as these are the "barba-pappas" among the distributions. 
Lets print the summary of detected distributions with the sum of square scores.

```python
dist.plot_summary()
```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1_summary.png" width="600" />
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
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1_prediction.png" width="600" />
</p>

The results of the prediction are stored in ``y_proba`` and ``y_pred``
```python

# Show the predictions for y
print(dist.y_pred)

# Show the probabilities for y that belong with the predictions
print(dist.y_proba)

# All predicted information is also stored in a structured dataframe
print(dist.df)
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
