# distfit

[![Python](https://img.shields.io/pypi/pyversions/distfit)](https://img.shields.io/pypi/pyversions/distfit)
[![PyPI Version](https://img.shields.io/pypi/v/distfit)](https://pypi.org/project/distfit/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/distfit/blob/master/LICENSE)

* Python package for probability distribution fitting and hypothesis testing.
* Probability distribution fitting is the fitting of a probability distribution to a series of data concerning the repeated measurement of a variable phenomenon.
* distfit scores each of the 89 different distributions for the fit wih the emperical distribution and return the best scoring distribution.


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
* It is advisable to create a new environment. 
```python
conda create -n env_distfit python=3.6
conda activate env_distfit
pip install numpy pandas tqdm matplotlib
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

## Import distfit package
```python
import distfit as distfit
```

## Example: Structure Learning
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/distfit/data/example_data.csv')
model = distfit.structure_learning(df)
G = distfit.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/distfit/blob/master/docs/figs/fig1.png" width="600" />
  
</p>

* Choosing various methodtypes and scoringtypes:
```python
model_hc_bic  = distfit.structure_learning(df, methodtype='hc', scoretype='bic')
```

#### df looks like this:
```
     Cloudy  Sprinkler  Rain  Wet_Grass
0         0          1     0          1
1         1          1     1          1
2         1          0     1          1
3         0          0     1          1
4         1          0     1          1
..      ...        ...   ...        ...
995       0          0     0          0
996       1          0     0          0
997       0          0     1          0
998       1          1     0          1
999       1          0     1          1
```


## Citation
Please cite distfit in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019distfit,
  title={distfit},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/distfit}},
}
```

## References
* http://pgmpy.org
* https://programtalk.com/python-examples/pgmpy.factors.discrete.TabularCPD/
* http://www.distfit.com/
* http://www.distfit.com/bnrepository/
   
## Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

## Contribute
* Contributions are welcome.

## © Copyright
See [LICENSE](LICENSE) for details.
