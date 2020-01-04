""" P = hypotesting(data,dataNull): Provides P-values for all values in data based on 
    the underlying null-distribution from dataNull. The emperical distribution 
    of dataNull is used to estimate the loc/scale/arg paramters for a theoretical 
    distirbution.
        
	P= hypotesting(data, background, <optional>)

 INPUT:
   data:           numpy array of numerical values
                   The iput data contains values for hypothesis testing.

   dataNull:       numpy array of numerical values
                   Background data: Null distribution which is used to compute Pvalues for the inputdata data.

 OPTIONAL

   distribution=   String: Set the distribution to use
                   'auto_small': (default) A smaller set of distributions: [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm]
                   'auto_full' : The full set of distributions
                   'norm'      : normal distribution
                   't'         : Students T distribution
                   etc

   bound=          String: Set whether you want returned a P-value for the lower/upper bounds or both
                   'both': Both (default)
                   'up':   Upperbounds
                   'down': Lowerbounds

   alpha=          Double : [0..1] Significance alpha
                   [0.05]: Default
   
   multtest=       [String]:
                   None             : No multiple Test (default)
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
    
    
   bins=           [Integer]: Bin size to make the estimation
                   [50]: (default)
   
   showfig=        [Boolean] [0,1]: Show figure
                   [0]: No
                   [1]: Yes (Default)

   outdist=        Dict: first parameter of the distfit function.
                   [outdist,_]=distfit(..)

   verbose   : Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes

 OUTPUT
	output


 EXAMPLE
   import distfit as distfit

   dataNull=np.random.normal(0, 2, 1000)
   data=[-8,-6,0,1,2,3,4,5,6,7,8,9,10]

   Pout = distfit.proba_parametric(data,dataNull)
   Pout = distfit.proba_parametric(data,dataNull, bound='down')
   Pout = distfit.proba_parametric(data,dataNull, bound='up')
   Pout = distfit.proba_parametric(data)

   Pout = distfit.proba_emperical(data,dataNull)
   Pout = distfit.proba_emperical(data)


   # In jupyter notebooks, its recommended to use showfig=2
   Pout = hypotesting(data,dataNull,showfig=2)

   
"""
#print(__doc__)
#https://people.duke.edu/~ccc14/sta-663/ResamplingAndMonteCarloSimulations.html

# Name        : hypotesting.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Oct. 2017
# Last edit   : April 2019

#%% Libraries
import numpy as np
import statsmodels.stats.multitest as multitest
import distfit as distfit
import matplotlib.pyplot as plt

#%% Emperical test
def proba_emperical(data, dataNull=None, alpha=0.05, bins=50, bound='both', multtest='fdr_bh', showfig=True, verbose=3):
    '''
    Suppose you have 2 data sets from unknown distribution and you want to test 
    if some arbitrary statistic (e.g 7th percentile) is the same in the 2 data sets - what can you do?
    An appropirate test statistic is the difference between the 7th percentile, 
    and if we knew the null distribution of this statisic, we could test for the 
    null hypothesis that the statistic = 0. 
    Permuting the labels of the 2 data sets allows us to create the empirical null distribution.
    '''
    args =dict()
    args['verbose']      = verbose
    args['showfig']      = showfig
    args['bound']        = bound
    args['alpha']        = alpha
    args['multtest']     = multtest
    args['bins']         = bins
 
    ciilow   = (0+(alpha/2))*100
    ciihigh = (1-(alpha/2))*100

    if isinstance(dataNull, type(None)):
        dataNull=data

    [n1, n2] = map(len, (data, dataNull))
    reps = 10000
    dataC = np.concatenate([data, dataNull])
    ps = np.array([np.random.permutation(n1+n2) for i in range(reps)])

    xp = dataC[ps[:, :n1]]
    yp = dataC[ps[:, n1:]]
    samples = np.percentile(xp, 7, axis=1) - np.percentile(yp, 7, axis=1)
    
    cii_low=np.percentile(samples, ciilow)
    cii_high=np.percentile(samples, ciihigh)

    teststat=np.ones_like(data)*np.nan
    Praw=np.ones_like(data)*np.nan
    for i in range(0,len(data)):
        getstat = np.percentile(data[i], 7) - np.percentile(dataNull, 7)
        getP=(2*np.sum(samples >= np.abs(getstat))/reps)
        getP=np.clip(getP,0,1)
#        if np.sign(getstat)<0:
#            getP=1-getP

        Praw[i] = getP
        teststat[i] = getstat
        if verbose>=4: print("[%.0f] - p-value = %f" %(data[i], getP))

    # Set bounds
    getbound = np.repeat('none',len(data))
    getbound[teststat>=cii_high]='up'
    getbound[teststat<=cii_low]='down'

    Padj = do_multtest(Praw, args['multtest'], verbose=args['verbose'])
    makefig(samples, teststat, Padj, cii_low, cii_high, args['alpha'], args['showfig'])
    
    out=dict()
    out['Praw']=Praw
    out['Padj']=Padj
    out['bound']=getbound
    out['cii_low']=cii_low
    out['cii_high']=cii_high
    
    return(out)
    
#%% Parametric tests
def proba_parametric(data, dataNull=[], distribution='auto_small', bound='both', alpha=0.05, multtest='fdr_bh', showfig=2, outdist=[], bins=50, verbose=3):
	# DECLARATIONS
    if 'list' in str(type(data)): data=np.array(data)
    if 'float' in str(type(data)): data=np.array([data])
    assert 'numpy.ndarray' in str(type(data)), 'data should be of type np.array or list'
    
    Praw = []
    Padj = []
    Param =dict()
    Param['verbose']      = verbose
    Param['showfig']      = showfig
    Param['distribution'] = distribution
    Param['bound']        = bound
    Param['alpha']        = alpha
    Param['multtest']     = multtest
    Param['bins']         = bins
        
    # Check which distribution fits best to the data
    if Param['verbose']>=3: print('[HYPOTHESIS TESTING] Analyzing underlying data distribution...')
    
    #
    if dataNull==[] and len(outdist)==0:
        if Param['verbose']>=3: print('[HYPOTHESIS TESTING] WARNING: Background distribution was absent, input data is used instead!')
        dataNull=np.array(data.copy())

    # Compute null-distribution parameters
    # [outdist, distname]  = distfit.parametric(dataNull, bins=50, distribution='norm', showfig=Param['showfig'], verbose=Param['verbose'])
    if len(outdist)==0:
        model  = distfit.parametric(dataNull, bins=Param['bins'], distribution=Param['distribution'], alpha=Param['alpha'], bound=Param['bound'], verbose=Param['verbose'])
        outdist = model['model']
        # distresults = model['model']
        # Plot
        distfit.plot(model, showfig=Param['showfig'])
    else:
        if Param['verbose']>=3: print('[HYPOTHESIS TESTING] Using existing fit.')

    # Get distribution and the parameters
    #dist  = getattr(st, outdist['name'])
    dist  = outdist['distribution']
    arg   = outdist['params'][:-2]
    loc   = outdist['params'][-2]
    scale = outdist['params'][-1]

    # Compute P-value for data based on null-distribution
    getP = dist.cdf(data, *arg, loc, scale) if arg else dist.pdf(data, loc, scale)
    #P = dist.pdf(data, *arg, loc, scale) if arg else dist.pdf(data, loc, scale)

    # Determine P based on upper/lower/no bounds
    if Param['bound']=='up':
        Praw  = 1-getP
    elif Param['bound']=='down':
        Praw = getP
    elif Param['bound']=='both':
        Praw = np.min([1-getP,getP], axis=0)
    else:
        if Param['verbose']>=3: print('[HYPOTHESIS TESTING] WARNING: "bounds" is not set correctly! Options are: up/down/both.')
        Praw=[]

    getbound = np.repeat('none',len(data))
    if not isinstance(outdist['CII_max_alpha'], type(None)):
        getbound[data>=outdist['CII_max_alpha']]='up'
    if not isinstance(outdist['CII_min_alpha'], type(None)):
        getbound[data<=outdist['CII_min_alpha']]='down'

    # Set all values in range[0..1]
    Praw = np.clip(Praw,0,1)

    # Multiple test correction
    Padj = do_multtest(Praw, Param['multtest'], verbose=Param['verbose'])

    # Add significant hits as line into the plot
    if Param['showfig']>0:
        if Param['verbose']>=3: print('[HYPOTHESIS TESTING] Making figure.')
        ax=outdist['ax']
        if not isinstance(ax, type(None)):
        # Plot only significant hits
            idx=np.where(Padj<=Param['alpha'])[0]
            for i in range(0,len(idx)):
                ax.axvline(x=data[idx[i]], ymin=0, ymax=1, linewidth=2, color='g', linestyle='--', alpha=0.8)

    # Make QQ-plot
    #http://www.statsmodels.org/dev/generated/statsmodels.graphics.gofplots.qqplot.html
    
#    http://matplotlib.org/mpl-probscale/tutorial/closer_look_at_viz.html
#    from matplotlib import pyplot
#    import probscale
#
#    getdist = dist(*outdist['params'])
#    
#    get_params = dist.fit(data, floc=0)
#    get_params_dist = dist(*get_params)
#    [fig, ax] = pyplot.subplots(figsize=(5, 5))
#    ax.set_aspect('equal')
#    common_opts = dict(plottype='qq', probax='x', problabel='Theoretical Quantiles', datalabel='Emperical Quantiles', scatter_kws=dict(label='Values') )
#
#    fig = probscale.probplot(data, ax=ax, dist=get_params_dist, **common_opts)
#    
#    limits = [np.min([ax.get_xlim(), ax.get_ylim()]),np.max([ax.get_xlim(), ax.get_ylim()]),]
#    ax.set_xlim(limits)
#    ax.set_ylim(limits)
#    ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0, label='Fit Lognormal Distribution')
#    ax.legend(loc='lower right')
#    seaborn.despine()
#    
#
#
#    
#    import statsmodels.api as sm
#    dataNullc = sm.add_constant(dataNull)
#    mod_fit = sm.OLS(data, dataNullc).fit()
#    res = mod_fit.resid # residuals
#    fig = sm.qqplot(res, line='45')
#    plt.show()
#    fig = sm.qqplot(res, dist, distargs=arg, loc=loc, scale=scale, line='45')
    
    # Distribution properties
#    https://docs.scipy.org/doc/scipy-0.19.1/reference/stats.html#module-scipy.stats
#    if len(fitparam)==3:
#        a     = fitparam[0]
#        loc   = fitparam[1]
#        scale = fitparam[2]
#    elif len(fitparam)==2:
#        loc   = fitparam[0]
#        scale = fitparam[1]
#    #end
#
#    P=stats.gammacpdf(4,a,loc,scale)
#    P=stats.gamma.cdf([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],a,loc,scale)

#    P=stats.norm.cdf(data, loc, scale)
#    P=stats.norm.cdf([0,1,2,3], loc, scale)
#    P=stats.norm.cdf([4,5,6,7,8], loc, scale)
#    P=stats.norm.cdf([9,10,11,12,14],loc, scale)
    
    #
    out=dict()
    out['Praw']=Praw
    out['Padj']=Padj
    out['bound']=getbound
    out['cii_low']=outdist['CII_min_alpha']
    out['cii_high']=outdist['CII_max_alpha']

#    outdist['CII_min_alpha']
#    outdist['CII_max_alpha']

    return(out)

#%%
def makefig(samples, teststat, Padj, cii_low, cii_high, alpha, showfig):
    if showfig:
        plt.subplots(figsize=(25,8))
        plt.hist(samples, 25, histtype='step')
        plt.axvline(cii_low, linestyle='--', c='r', label='cii low')
        plt.axvline(cii_high, linestyle='--', c='r', label='cii high')
        for i in range(0,len(teststat)):
            if Padj[i]<=alpha:
                plt.axvline(teststat[i], c='g')
    
#%%
def do_multtest(Praw, multtest='fdr_bh', verbose=3):
    if verbose>=3: print("[HYPOTHESIS TESTING] Multiple test correction..[%s]" %multtest)
    if not isinstance(multtest, type(None)):
        Padj = multitest.multipletests(Praw, method=multtest)
        Padj = Padj[1]
    else:
        Padj=Praw
    
    Padj = np.clip(Padj,0,1)
    return(Padj)