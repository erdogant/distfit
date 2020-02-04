"""Compute probability based on theoretical distribution.

   description
   -----------
   Provides P-values for all values in data based on the underlying null-distribution
   from dataNull. The emperical distribution of dataNull is used to estimate
   the loc/scale/arg paramters for a theoretical distirbution.


    import hypotesting as hypo

    hypo.proba_emperical(X)
    hypo.proba_parametric(X)

"""

# Name        : hypotesting.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See Licences


# %% Libraries
import numpy as np
import statsmodels.stats.multitest as multitest
import distfit as dist
import pandas as pd


# %% Emperical test
def proba_emperical(data, dataNull=None, alpha=0.05, bins=50, bound='both', multtest='fdr_bh', verbose=3):
    """Compute Probability based on an emperical test.

    Description
    -----------
    Suppose you have 2 data sets from unknown distribution and you want to test
    if some arbitrary statistic (e.g 7th percentile) is the same in the 2 data sets.
    An appropirate test statistic is the difference between the 7th percentile,
    and if we knew the null distribution of this statisic, we could test for the
    null hypothesis that the statistic = 0. Permuting the labels of the 2 data sets
    allows us to create the empirical null distribution.


    Parameters
    ----------
    data : Numpy array: Emperical data.

    dataNull : numpy array, optional (default: None)
        Background data: Null distribution which is used to compute Pvalues for the inputdata data.

    alpha : Float [0-1], optional (default: 0.05)
        Significance alpha.

    bins : Int, optional, (default: 50)
        Bin size to determine the emperical historgram.

    bound : String, optional (default: 'both')
        Set whether you want returned a P-value for the lower/upper bounds or both.
        'both': Both (default)
        'up':   Upperbounds
        'low': Lowerbounds

    multtest : String, optional (default: 'fdr_bh')
        Multiple testing method.
        None             : No multiple Test
        'bonferroni'     : one-step correction
        'sidak'          : one-step correction
        'holm-sidak'     : step down method using Sidak adjustments
        'holm'           : step-down method using Bonferroni adjustments
        'simes-hochberg' : step-up method  (independent)
        'hommel'         : closed method based on Simes tests (non-negative)
        'fdr_bh'         : Benjamini/Hochberg  (non-negative)
        'fdr_by'         : Benjamini/Yekutieli (negative)
        'fdr_tsbh'       : two stage fdr correction (non-negative)
        'fdr_tsbky'      : two stage fdr correction (non-negative)

    verbose : Int [1-5], optional (default: 3)
        Print information to screen.

    Returns
    -------
    dict.

    """
    args = dict()
    args['verbose'] = verbose
    args['bound'] = bound
    args['alpha'] = alpha
    args['multtest'] = multtest
    args['bins'] = bins

    ciilow = (0 + (alpha / 2)) * 100
    ciihigh = (1 - (alpha / 2)) * 100

    if isinstance(dataNull, type(None)):
        dataNull=data

    [n1, n2] = map(len, (data, dataNull))
    reps = 10000
    dataC = np.concatenate([data, dataNull])
    ps = np.array([np.random.permutation(n1 + n2) for i in range(reps)])

    xp = dataC[ps[:, :n1]]
    yp = dataC[ps[:, n1:]]
    samples = np.percentile(xp, 7, axis=1) - np.percentile(yp, 7, axis=1)

    cii_low=np.percentile(samples, ciilow)
    cii_high=np.percentile(samples, ciihigh)

    teststat=np.ones_like(data) * np.nan
    Praw=np.ones_like(data) * np.nan
    for i in range(0,len(data)):
        getstat = np.percentile(data[i], 7) - np.percentile(dataNull, 7)
        getP=(2 * np.sum(samples >= np.abs(getstat)) / reps)
        getP=np.clip(getP,0,1)
        Praw[i] = getP
        teststat[i] = getstat
        if verbose>=4: print("[%.0f] - p-value = %f" %(data[i], getP))

    # Set bounds
    getbound = np.repeat('none',len(data))
    getbound[teststat>=cii_high]='up'
    getbound[teststat<=cii_low]='low'

    # Compute multiple testing to correct for Pvalues
    Padj = _do_multtest(Praw, args['multtest'], verbose=args['verbose'])

    out=dict()
    out['method']='emperical'
    out['Praw']=Praw
    out['Padj']=Padj
    out['bound']=getbound
    out['cii_low']=cii_low
    out['cii_high']=cii_high
    out['alpha']=args['alpha']
    out['teststat']=teststat
    out['samples']=samples

    return(out)


# %% Parametric tests
def proba_parametric(data, dataNull=[], alpha=0.05, bins=50, bound='both', multtest='fdr_bh', distribution='auto_small', model=None, verbose=3):
    """Compute Probability based on an parametric test.

    Parameters
    ----------
    data : Numpy array: Emperical data.

    dataNull : numpy array, optional (default: None)
        Background data: Null distribution which is used to compute Pvalues for the input data data.

    model : dict, optional (default: None)
        The model created by the .fit() function.

    distribution : String, (default:'auto_small')
        The (set) of distribution to test.
        'auto_small': A smaller set of distributions: [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm]
        'auto_full' : The full set of distributions
        'norm'      : normal distribution
        't'         : Students T distribution
        etc

    bound : String, optional (default: 'both')
        Set whether you want returned a P-value for the lower/upper bounds or both.
        'both': Both (default)
        'up':   Upperbounds
        'low': Lowerbounds

    alpha : Float [0-1], optional (default: 0.05)
        Significance alpha.

    multtest : String, optional (default: 'fdr_bh')
        Multiple testing method.
        None             : No multiple Test
        'bonferroni'     : one-step correction
        'sidak'          : one-step correction
        'holm-sidak'     : step down method using Sidak adjustments
        'holm'           : step-down method using Bonferroni adjustments
        'simes-hochberg' : step-up method  (independent)
        'hommel'         : closed method based on Simes tests (non-negative)
        'fdr_bh'         : Benjamini/Hochberg  (non-negative)
        'fdr_by'         : Benjamini/Yekutieli (negative)
        'fdr_tsbh'       : two stage fdr correction (non-negative)
        'fdr_tsbky'      : two stage fdr correction (non-negative)

    bins : Int, optional, (default: 50)
        Bin size to determine the emperical historgram.

    verbose : Int [1-5], optional (default: 3)
        Print information to screen.

    Returns
    -------
    None.

    """
    if 'list' in str(type(data)): data=np.array(data)
    if 'float' in str(type(data)): data=np.array([data])
    assert 'numpy.ndarray' in str(type(data)), 'data should be of type np.array or list'
    # if alpha==None: alpha=1

    Param = dict()
    Param['verbose'] = verbose
    Param['distribution'] = distribution
    Param['bound'] = bound
    Param['alpha'] = alpha
    Param['multtest'] = multtest
    Param['bins'] = bins

    # Check which distribution fits best to the data
    if Param['verbose']>=3: print('[DISTFIT.proba] Analyzing underlying data distribution...')

    if dataNull==[] and model is None:
        if Param['verbose']>=3: print('[DISTFIT.proba] WARNING: Background distribution was absent, input data is used instead!')
        dataNull=np.array(data.copy())

    # Compute null-distribution parameters
    if (model is None) or model['Param']['alpha']!=Param['alpha']:
        model = dist.fit(dataNull, bins=Param['bins'], distribution=Param['distribution'], alpha=Param['alpha'], bound=Param['bound'], verbose=Param['verbose'])
    else:
        if Param['verbose']>=3: print('[DISTFIT.proba] Using existing fit.')

    # Get distribution and the parameters
    # dist  = getattr(st, model['model']['name'])
    getdist = model['model']['distribution']
    arg = model['model']['params'][:-2]
    loc = model['model']['params'][-2]
    scale = model['model']['params'][-1]

    # Compute P-value for data based on null-distribution
    getP = getdist.cdf(data, *arg, loc, scale) if arg else getdist.pdf(data, loc, scale)

    # Determine P based on upper/lower/no bounds
    if Param['bound']=='up' or Param['bound']=='right':
        Praw = 1 - getP
    elif Param['bound']=='low' or Param['bound']=='left':
        Praw = getP
    elif Param['bound']=='both':
        Praw = np.min([1 - getP, getP], axis=0)
    else:
        if Param['verbose']>=3: print('[DISTFIT.proba] WARNING: "bounds" is not set correctly! Options are: up/down/both.')
        Praw=[]

    # Set all values in range[0..1]
    Praw = np.clip(Praw,0,1)
    # Multiple test correction
    Padj = _do_multtest(Praw, Param['multtest'], verbose=Param['verbose'])
    # up/down based on threshold
    getbound = np.repeat('none', len(data))
    if (Param['alpha'] is None):
        Param['alpha']=1
    if not isinstance(model['model']['CII_max_alpha'], type(None)):
        getbound[data>=model['model']['CII_max_alpha']]='up'
    if not isinstance(model['model']['CII_min_alpha'], type(None)):
        getbound[data<=model['model']['CII_min_alpha']]='low'

    # Make structured output
    df = pd.DataFrame()
    df['data'] = data
    df['P'] = Praw
    df['Padj'] = Padj
    df['bound'] = getbound

    # Return
    out = model
    out['method'] = 'parametric'
    out['proba'] = df
    return(out)


# %%
def _do_multtest(Praw, multtest='fdr_bh', verbose=3):
    if not isinstance(multtest, type(None)):
        if verbose>=3: print("[DISTFIT.proba] Multiple test correction..[%s]" %multtest)
        Padj = multitest.multipletests(Praw, method=multtest)[1]
    else:
        Padj=Praw

    Padj = np.clip(Padj, 0, 1)
    return(Padj)
