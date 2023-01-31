import numpy as np
from distfit import distfit
import unittest
from scipy.stats import binom

def show_figures(dist):
    charts = ['PDF', 'CDF']
    n_top=[1, 10]
    for chart in charts:
        for n in n_top:
            dist.plot(chart=chart, n_top=n)
            dist.plot(chart=chart, n_top=n, cii_properties=None, emp_properties={}, pdf_properties={}, bar_properties={})
            dist.plot(chart=chart, n_top=n, cii_properties={}, emp_properties=None, pdf_properties={}, bar_properties={})
            dist.plot(chart=chart, n_top=n, cii_properties={}, emp_properties={}, pdf_properties=None, bar_properties={})
            dist.plot(chart=chart, n_top=n, cii_properties={}, emp_properties={}, pdf_properties={}, bar_properties=None)
    dist.plot_summary()
    
class Test_DISTFIT(unittest.TestCase):

    def test_figures(self):
        from distfit import distfit
        dist = distfit()
        X = binom(8, 0.5).rvs(1000)
        dist = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
        dist.fit_transform(X, verbose='info');
        show_figures(dist)

        X = np.random.uniform(0, 1000, 10000)
        dist = distfit(distr='uniform')
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        results = dist.fit_transform(X)
        dist.predict(y)
        show_figures(dist)
        
        X = np.random.exponential(0.5, 10000)
        dist = distfit(distr='expon')
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        results = dist.fit_transform(X)
        dist.plot(figsize=(15, 12), grid=False)
        dist.predict(y)
        show_figures(dist)
        
        X = np.random.normal(0, 2, 10000)
        dist = distfit(distr='norm')
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        results = dist.fit_transform(X)
        dist.predict(y)
        show_figures(dist)
        
        dist.plot(bar_properties={'color': '#808080', 'label': None},
                  pdf_properties={'color': 'r'},
                  emp_properties={'color': '#000000', 'linewidth': 3},
                  cii_properties={'color': 'b'})


    def test_distfit(self):
        from distfit import distfit

        X = np.random.normal(0, 2, 1000)
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        # Initialize
        dist = distfit()
        assert np.all(np.isin(['method', 'alpha', 'bins', 'distr', 'multtest', 'n_perm'], dir(dist)))
        # Fit and transform data
        dist.fit_transform(X, verbose='info')
    
        # TEST 1: check output is unchanged
        assert np.all(np.isin(['method', 'model', 'summary', 'histdata', 'size'], dir(dist)))
        # TEST 2: Check model output is unchanged
        assert [*dist.model.keys()]==['distr', 'stats', 'params', 'name', 'model', 'score', 'loc', 'scale', 'arg', 'CII_min_alpha', 'CII_max_alpha']
    
        # TEST 3: Check specific distribution
        dist = distfit(distr='t')
        dist.fit_transform(X)
        assert dist.model['name']=='t'
    
        # TEST 4: Check specific distribution
        dist = distfit(distr='t', alpha=None)
        dist.fit_transform(X)
        assert dist.model['CII_min_alpha'] is not None
        assert dist.model['CII_max_alpha'] is not None
    
        # TEST 4A: Check multiple distribution
        dist = distfit(distr=['norm', 't', 'gamma'])
        results = dist.fit_transform(X)
        assert np.all(np.isin(results['summary']['distr'].values, ['gamma', 't', 'norm']))
    
        # TEST 5: Bound check
        dist = distfit(distr='t', bound='up', alpha=0.05)
        dist.fit_transform(X, verbose=0)
        assert dist.model['CII_min_alpha'] is None
        assert dist.model['CII_max_alpha'] is not None
        dist = distfit(distr='t', bound='down', alpha=0.05)
        dist.fit_transform(X, verbose=0)
        assert dist.model['CII_min_alpha'] is not None
        assert dist.model['CII_max_alpha'] is None
        dist = distfit(distr='t', bound='both', alpha=0.05)
        dist.fit_transform(X, verbose=0)
        assert dist.model['CII_min_alpha'] is not None
        assert dist.model['CII_max_alpha'] is not None
    
        # TEST 6: Distribution check: Make sure the right loc and scale paramters are detected
        X = np.random.normal(0, 2, 10000)
        dist = distfit(distr='norm', alpha=0.05)
        dist.fit_transform(X, verbose=0)
        dist.model['loc']
        '%.1f' %dist.model['scale']=='2.0'
        '%.1f' %np.abs(dist.model['loc'])=='0.0'
    
        # TEST 7
        X = np.random.normal(0, 2, 1000)
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
    
        # TEST 1: Check bounds
        out1 = distfit(distr='norm',  bound='up')
        out1.fit_transform(X, verbose=0)
        out1.predict(y, verbose=0)
        assert np.all(np.isin(np.unique(out1.results['y_pred']), ['none','up']))
    
        out2 = distfit(distr='norm',  bound='down')
        out2.fit_transform(X, verbose=0)
        out2.predict(y, verbose=0)
        assert np.all(np.isin(np.unique(out2.results['y_pred']), ['none','down']))
    
        out3 = distfit(distr='norm',  bound='down')
        out3.fit_transform(X, verbose=0)
        out3.predict(y, verbose=0)
        assert np.all(np.isin(np.unique(out3.results['y_pred']), ['none','down','up']))
    
        # TEST 8: Check different sizes array
        X = np.random.normal(0, 2, [10,100])
        dist = distfit(distr='norm',  bound='up')
        dist.fit_transform(X, verbose=0)
        dist.predict(y, verbose=0)
        assert np.all(np.isin(np.unique(dist.results['y_pred']), ['none','up']))
    
        # TEST 9
        data_random = np.random.normal(0, 2, 1000)
        data = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        dist = distfit()
        dist.fit_transform(X, verbose=0)
    
        # TEST 10 Check number of output probabilities
        dist.fit_transform(X, verbose=0)
        dist.predict(y)
        assert dist.results['y_proba'].shape[0]==len(y)
    
        # TEST 11: Check whether alpha responds on results
        out1 = distfit(alpha=0.05)
        out1.fit_transform(X, verbose=0)
        out1.predict(y)
    
        out2 = distfit(alpha=0.2)
        out2.fit_transform(X, verbose=0)
        out2.predict(y)
    
        assert np.all(out1.y_proba==out2.y_proba)
        assert not np.all(out1.results['y_pred']==out2.results['y_pred'])
        assert np.all(out1.results['P']==out2.results['P'])
        assert sum(out1.results['y_pred']=='none')>sum(out2.results['y_pred']=='none')
    
        # TEST 12: Check different sizes array
        X = np.random.normal(0, 2, [10,100])
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
    
        dist = distfit(bound='up')
        dist.fit_transform(X, verbose=0)
        dist.predict(y)
        assert np.all(np.isin(np.unique(dist.results['y_pred']), ['none','up']))
    
        # TEST 13: Precentile
        X = np.random.normal(0, 2, [10,100])
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        dist = distfit(method='percentile')
        dist.fit_transform(X, verbose=0)
        results=dist.predict(y)
        assert np.all(np.isin([*results.keys()], ['y', 'y_proba', 'y_pred', 'P', 'teststat']))
    
        # TEST 14: Quantile
        dist = distfit(method='quantile')
        dist.fit_transform(X, verbose=0)
        results=dist.predict(y)
        assert np.all(np.isin([*results.keys()],  ['y', 'y_proba', 'y_pred', 'teststat']))
    
        # TEST 15: Discrete
        import random
        random.seed(10)
        from scipy.stats import binom
        # Generate random numbers
        X = binom(8, 0.5).rvs(10000)
    
        dist = distfit(method='discrete', f=1.5, weighted=True)
        dist.fit_transform(X, verbose='info')
        assert dist.model['n']==8
        assert np.round(dist.model['p'], decimals=1)==0.5
    
        # check output is unchanged
        assert np.all(np.isin(['method', 'model', 'summary', 'histdata', 'size'], dir(dist)))
        # TEST 15A
        assert [*dist.model.keys()]==['name', 'distr', 'model', 'params', 'score', 'chi2r', 'n', 'p', 'CII_min_alpha', 'CII_max_alpha']
        # TEST 15B
        results = dist.predict([0, 1, 10, 11, 12])
        assert np.all(np.isin([*results.keys()], ['y', 'y_proba', 'y_pred', 'P', 'y_bool']))

        from distfit import distfit
        # Set parameters for the test-case
        X = binom(8, 0.5).rvs(10000)
        # Initialize distfit for discrete distribution for which the binomial distribution is used. 
        dist = distfit(method='discrete')
        # Run distfit to and determine whether we can find the parameters from the data.
        results = dist.fit_transform(X)
        # Get the model and best fitted parameters.
        y = [0, 1, 10, 11, 12]
        results = dist.predict(y)
        dist.plot()
        
