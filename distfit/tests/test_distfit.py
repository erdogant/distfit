import numpy as np
from distfit import distfit
import unittest
from scipy.stats import binom

def show_figures(dfit):
    charts = ['PDF', 'CDF']
    n_top=[1, 10]
    for chart in charts:
        for n in n_top:
            dfit.plot(chart=chart, n_top=n)
            dfit.plot(chart=chart, n_top=n, cii_properties=None, emp_properties={}, pdf_properties={}, bar_properties={})
            dfit.plot(chart=chart, n_top=n, cii_properties={}, emp_properties=None, pdf_properties={}, bar_properties={})
            dfit.plot(chart=chart, n_top=n, cii_properties={}, emp_properties={}, pdf_properties=None, bar_properties={})
            dfit.plot(chart=chart, n_top=n, cii_properties={}, emp_properties={}, pdf_properties={}, bar_properties=None)
    dfit.plot_summary()
    
class Test_DISTFIT(unittest.TestCase):

    def test_figures(self):
        from distfit import distfit
        dfit = distfit()
        X = binom(8, 0.5).rvs(1000)
        dfit = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
        dfit.fit_transform(X, verbose='info');
        show_figures(dfit)

        X = np.random.uniform(0, 1000, 10000)
        dfit = distfit(distr='uniform')
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        results = dfit.fit_transform(X)
        dfit.predict(y)
        show_figures(dfit)
        
        X = np.random.exponential(0.5, 10000)
        dfit = distfit(distr='expon')
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        results = dfit.fit_transform(X)
        dfit.plot(figsize=(15, 12), grid=False)
        dfit.predict(y)
        show_figures(dfit)
        
        X = np.random.normal(0, 2, 10000)
        dfit = distfit(distr='norm')
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        results = dfit.fit_transform(X)
        dfit.predict(y)
        show_figures(dfit)
        
        dfit.plot(bar_properties={'color': '#808080', 'label': None},
                  pdf_properties={'color': 'r'},
                  emp_properties={'color': '#000000', 'linewidth': 3},
                  cii_properties={'color': 'b'})


    def test_distfit(self):
        from distfit import distfit

        X = np.random.normal(0, 2, 1000)
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        # Initialize
        dfit = distfit()
        assert np.all(np.isin(['method', 'alpha', 'bins', 'name', 'multtest', 'n_perm'], dir(dfit)))
        # Fit and transform data
        dfit.fit_transform(X, verbose='info')
    
        # TEST 1: check output is unchanged
        assert np.all(np.isin(['method', 'model', 'summary', 'histdata', 'size'], dir(dfit)))
        # TEST 2: Check model output is unchanged
        assert [*dfit.model.keys()]==['name', 'stats', 'params', 'name', 'model', 'score', 'loc', 'scale', 'arg', 'CII_min_alpha', 'CII_max_alpha']
    
        # TEST 3: Check specific distribution
        dfit = distfit(distr='t')
        dfit.fit_transform(X)
        assert dfit.model['name']=='t'
    
        # TEST 4: Check specific distribution
        dfit = distfit(distr='t', alpha=None)
        dfit.fit_transform(X)
        assert dfit.model['CII_min_alpha'] is not None
        assert dfit.model['CII_max_alpha'] is not None
    
        # TEST 4A: Check multiple distribution
        dfit = distfit(distr=['norm', 't', 'gamma'])
        results = dfit.fit_transform(X)
        assert np.all(np.isin(results['summary']['name'].values, ['gamma', 't', 'norm']))
    
        # TEST 5: Bound check
        dfit = distfit(distr='t', bound='up', alpha=0.05)
        dfit.fit_transform(X, verbose=0)
        assert dfit.model['CII_min_alpha'] is None
        assert dfit.model['CII_max_alpha'] is not None
        dfit = distfit(distr='t', bound='down', alpha=0.05)
        dfit.fit_transform(X, verbose=0)
        assert dfit.model['CII_min_alpha'] is not None
        assert dfit.model['CII_max_alpha'] is None
        dfit = distfit(distr='t', bound='both', alpha=0.05)
        dfit.fit_transform(X, verbose=0)
        assert dfit.model['CII_min_alpha'] is not None
        assert dfit.model['CII_max_alpha'] is not None
    
        # TEST 6: Distribution check: Make sure the right loc and scale paramters are detected
        X = np.random.normal(0, 2, 10000)
        dfit = distfit(distr='norm', alpha=0.05)
        dfit.fit_transform(X, verbose=0)
        dfit.model['loc']
        '%.1f' %dfit.model['scale']=='2.0'
        '%.1f' %np.abs(dfit.model['loc'])=='0.0'
    
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
        dfit = distfit(distr='norm',  bound='up')
        dfit.fit_transform(X, verbose=0)
        dfit.predict(y, verbose=0)
        assert np.all(np.isin(np.unique(dfit.results['y_pred']), ['none','up']))
    
        # TEST 9
        data_random = np.random.normal(0, 2, 1000)
        data = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        dfit = distfit()
        dfit.fit_transform(X, verbose=0)
    
        # TEST 10 Check number of output probabilities
        dfit.fit_transform(X, verbose=0)
        dfit.predict(y)
        assert dfit.results['y_proba'].shape[0]==len(y)
    
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
    
        dfit = distfit(bound='up')
        dfit.fit_transform(X, verbose=0)
        dfit.predict(y)
        assert np.all(np.isin(np.unique(dfit.results['y_pred']), ['none','up']))
    
        # TEST 13: Precentile
        X = np.random.normal(0, 2, [10,100])
        y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]
        dfit = distfit(method='percentile')
        dfit.fit_transform(X, verbose=0)
        results=dfit.predict(y)
        assert np.all(np.isin([*results.keys()], ['y', 'y_proba', 'y_pred', 'P', 'teststat']))
    
        # TEST 14: Quantile
        dfit = distfit(method='quantile')
        dfit.fit_transform(X, verbose=0)
        results=dfit.predict(y)
        assert np.all(np.isin([*results.keys()],  ['y', 'y_proba', 'y_pred', 'teststat']))
    
        # TEST 15: Discrete
        import random
        random.seed(10)
        from scipy.stats import binom
        # Generate random numbers
        X = binom(8, 0.5).rvs(10000)
    
        dfit = distfit(method='discrete', f=1.5, weighted=True)
        dfit.fit_transform(X, verbose='info')
        assert dfit.model['n']==8
        assert np.round(dfit.model['p'], decimals=1)==0.5
    
        # check output is unchanged
        assert np.all(np.isin(['method', 'model', 'summary', 'histdata', 'size'], dir(dfit)))
        # TEST 15A
        assert [*dfit.model.keys()]==['name', 'name', 'model', 'params', 'score', 'chi2r', 'n', 'p', 'CII_min_alpha', 'CII_max_alpha']
        # TEST 15B
        results = dfit.predict([0, 1, 10, 11, 12])
        assert np.all(np.isin([*results.keys()], ['y', 'y_proba', 'y_pred', 'P', 'y_bool']))

        from distfit import distfit
        # Set parameters for the test-case
        X = binom(8, 0.5).rvs(10000)
        # Initialize distfit for discrete distribution for which the binomial distribution is used. 
        dfit = distfit(method='discrete')
        # Run distfit to and determine whether we can find the parameters from the data.
        results = dfit.fit_transform(X)
        # Get the model and best fitted parameters.
        y = [0, 1, 10, 11, 12]
        results = dfit.predict(y)
        dfit.plot()
        
