"""Compute best fit to your emperical distribution for 89 different theoretical distributions using the Sum of Squared errors (SSE) estimates.

    import distfit as dist

    model  = dist.fit(X)
    fig,ax = dist.plot(model)
"""

# ---------------------------------
# Name        : distfit.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# ---------------------------------


# %% Libraries
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# %% Main
def fit(X, alpha=0.05, bins=50, bound='both', distribution='auto_small', verbose=3):
    """Fit best scoring theoretical distribution to the emperical data (X).

    Parameters
    ----------
    X : Numpy array: Emperical data.

    alpha : String, optional (default: 0.05)
        Significance alpha.

    bins : Int, optional, (default: 50)
        Bin size to determine the emperical historgram.

    bound : String, optional (default: 'both')
        Set whether you want returned a P-value for the lower/upper bounds or both.
        'both': Both (default)
        'up':   Upperbounds
        'low': Lowerbounds

    distribution : String, (default:'auto_small')
        The (set) of distribution to test.
        'auto_small': A smaller set of distributions: [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm]
        'auto_full' : The full set of distributions
        'norm'      : normal distribution
        't'         : Students T distribution
        etc

    verbose : Int [1-5], optional (default: 3)
        Print information to screen.

    Returns
    -------
    dict.

    """
    Param = {}
    Param['verbose'] = verbose
    Param['bins'] = bins
    Param['distribution'] = distribution
    Param['alpha'] = alpha
    Param['bound'] = bound
    assert len(X)>0, print('[DISTFIT] Error: Input X is empty!')

    # Format the X
    X = _format_data(X)

    # Get list of distributions to check
    DISTRIBUTIONS = _get_distributions(Param['distribution'])

    # Get histogram of original X
    [y_obs, X] = _get_hist_params(X, Param['bins'])

    # Compute best distribution fit on the emperical X
    out_summary, model = _compute_score_distribution(X, y_obs, X, DISTRIBUTIONS, verbose=Param['verbose'])

    # Determine confidence intervals on the best fitting distribution
    model = _compute_cii(model, alpha=Param['alpha'], bound=Param['bound'])

    # Return
    out = {}
    out['method'] = 'parametric'
    out['model'] = model
    out['summary'] = out_summary
    out['histdata'] = (y_obs, X)
    out['size'] = len(X)
    out['Param'] = Param
    return(out)


# %% Plot
def plot(model, title='', figsize=(10,8), xlim=None, ylim=None, verbose=3):
    """Make plot.

    Parameters
    ----------
    model : dict
        The model that is created by the .fit() function.

    title : String, optional (default: '')
        Title of the plot.

    figsize : tuple, optional (default: (10,8))
        The figure size.

    xlim : Float, optional (default: None)
        Limit figure in x-axis.

    ylim : Float, optional (default: None)
        Limit figure in y-axis.

    verbose : Int [1-5], optional (default: 3)
        Print information to screen.

    Returns
    -------
    fig : Figure
    ax : ax of Figure

    """
    if model['method']=='parametric':
        fig, ax = _plot_parametric(model, title=title, figsize=figsize, xlim=xlim, ylim=ylim, verbose=verbose)
    elif model['method']=='emperical':
        fig, ax = _plot_emperical(model, title=title, figsize=figsize, xlim=xlim, ylim=ylim, verbose=verbose)
    else:
        fig, ax = None, None

    return fig, ax


# %% Plot
def _plot_emperical(model, title='', figsize=(15,8), xlim=None, ylim=None, verbose=3):
    fig, ax = plt.subplots(figsize=figsize)
    plt.hist(model['samples'], 25, histtype='step', label='Emperical distribution')
    ax.axvline(model['cii_low'], linestyle='--', c='r', label='cii low')
    ax.axvline(model['cii_high'], linestyle='--', c='r', label='cii high')
    for i in range(0,len(model['teststat'])):
        if model['Padj'][i]<=model['alpha']:
            ax.axvline(model['teststat'][i], c='g')

    # Limit axis
    if xlim is not None:
        plt.xlim(xmin=xlim[0], xmax=xlim[1])
    if ylim is not None:
        plt.ylim(ymin=ylim[0], ymax=ylim[1])

    ax.grid(True)
    ax.set_title(title)
    ax.legend()
    return fig, ax


# %% Plot
def _plot_parametric(model, title='', figsize=(10,8), xlim=None, ylim=None, verbose=3):
    # Store output and function parameters
    out_dist = model['summary']
    out_dist = model['model']
    Param = model['Param']
    Param['title'] = title
    Param['figsize'] = figsize
    Param['xlim'] = xlim
    Param['ylim'] = ylim

    # Make figure
    best_dist = out_dist['distribution']
    best_fit_name = out_dist['name']
    best_fit_param = out_dist['params']
    arg = out_dist['params'][:-2]
    loc = out_dist['params'][-2]
    scale = out_dist['params'][-1]
    dist = getattr(st, out_dist['name'])

    # Plot line
    getmin = dist.ppf(0.0000001, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.0000001, loc=loc, scale=scale)
    getmax = dist.ppf(0.9999999, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.9999999, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(getmin, getmax, model['size'])
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    # ymax=max(model['histdata'][0])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(model['histdata'][1],model['histdata'][0], color='k', linewidth=1, label='Emperical distribution')
    ax.plot(x, y, 'b-', linewidth=1, label=best_fit_name)

    # Plot vertical line To stress the cut-off point
    if model['model']['CII_min_alpha'] is not None:
        label = 'CII low ' + '(' + str(Param['alpha']) + ')'
        ax.axvline(x=out_dist['CII_min_alpha'], ymin=0, ymax=1, linewidth=1.3, color='r', linestyle='dashed', label=label)

    if model['model']['CII_max_alpha'] is not None:
        label = 'CII high ' + '(' + str(Param['alpha']) + ')'
        ax.axvline(x=out_dist['CII_max_alpha'], ymin=0, ymax=1, linewidth=1.3, color='r', linestyle='dashed', label=label)

    # Make text for plot
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_param)])
    ax.set_title('%s\n%s\n%s' %(Param['title'], best_fit_name, param_str))
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

    # Limit axis
    if Param['xlim'] is not None:
        plt.xlim(xmin=Param['xlim'][0], xmax=Param['xlim'][1])
    if Param['ylim'] is not None:
        plt.ylim(ymin=Param['ylim'][0], ymax=Param['ylim'][1])

    # Add significant hits as line into the plot. This data is dervived from dist.proba_parametric
    if not isinstance(model.get('tests', None), type(None)):
        # Plot significant hits
        if Param['alpha'] is None:
            Param['alpha']=1
        idxIN=np.where(model['tests']['Padj'].values<=Param['alpha'])[0]
        if verbose>=3: print("[DISTFIT.plot] Number of significant regions detected: %d" %(len(idxIN)))
        for i in idxIN:
            ax.axvline(x=model['tests']['data'].values[i], ymin=0, ymax=1, linewidth=1, color='g', linestyle='--', alpha=0.8)

        # Plot the samples that are not signifcant after multiple test
        idxOUT=np.where(model['tests']['Padj'].values>Param['alpha'])[0]
        ax.scatter(model['tests']['data'].values[idxOUT], np.zeros(len(idxOUT)), color='red', marker='x', alpha=0.8, linewidth=1, label='Not significant')

    ax.legend()
    ax.grid(True)

    if Param['verbose']>=3: print("[DISTFIT.plot] Estimated distribution: %s [loc:%f, scale:%f]" %(out_dist['name'],out_dist['params'][-2],out_dist['params'][-1]))
    return (fig, ax)


# %% Utils
def _format_data(data):
    # Convert pandas to numpy
    if str(data.dtype)=='O': data=data.astype(float)
    if 'pandas' in str(type(data)): data = data.values
    # Make sure its a vector
    data = data.ravel()
    return(data)


# Get the distributions based on user input
def _get_distributions(distribution):
    DISTRIBUTIONS=[]
    # Distributions to check
    if distribution=='auto_full':
        DISTRIBUTIONS = [st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
                         st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
                         st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
                         st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
                         st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
                         st.invweibull,st.johnsonsb,st.johnsonsu,st.laplace,st.levy,st.levy_l,st.levy_stable,
                         st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,
                         st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
                         st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
                         st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
    elif distribution=='auto_small':
        DISTRIBUTIONS = [st.norm, st.expon, st.pareto, st.dweibull, st.t, st.genextreme, st.gamma, st.lognorm, st.beta, st.uniform]
    else:
        # Connect object with variable to be used as a function again.
        DISTRIBUTIONS = [getattr(st, distribution)]

    return(DISTRIBUTIONS)


# Get histogram of original data
def _get_hist_params(data, bins):
    [y_obs, X] = np.histogram(data, bins=bins, density=True)
    X = (X + np.roll(X, -1))[:-1] / 2.0
    return(y_obs, X)


# Compute score for each distribution
def _compute_score_distribution(data, y_obs, X, DISTRIBUTIONS, verbose=3):
    out = []
    out_dist = {}
    out_dist['distribution'] = st.norm
    out_dist['params'] = (0.0, 1.0)
    best_sse = np.inf
    out = pd.DataFrame(index=range(0,len(DISTRIBUTIONS)), columns=['Distribution','SSE','LLE','loc','scale','arg'])

    # Estimate distribution parameters
    for i,dist in enumerate(DISTRIBUTIONS):
        logLik=0

        # Try to fit the dist. However this can result in an error so therefore you need to try-catch
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                # fit dist to data
                params = dist.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = dist.pdf(X, loc=loc, scale=scale, *arg)
                # Compute SSE
                sse = np.sum(np.power(y_obs - pdf, 2.0))
                logLik = np.nan

                # try:
                #     logLik = -np.sum( dist.logpdf(y_obs, loc=loc, scale=scale) )
                # except Exception:
                #     pass
#                if len(params)>2:
#                    logLik = -np.sum( dist.logpdf(y_obs, arg=arg, loc=loc, scale=scale) )
#                else:
#                    logLik = -np.sum( dist.logpdf(y_obs, loc=loc, scale=scale) )

#                # Store results
                out.values[i,0] = dist.name
                out.values[i,1] = sse
                out.values[i,2] = logLik
                out.values[i,3] = loc
                out.values[i,4] = scale
                out.values[i,5] = arg

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_sse = sse
                    out_dist['name'] = dist.name
                    out_dist['distribution'] = dist
                    out_dist['params'] = params
                    out_dist['sse'] = sse
                    out_dist['loc'] = loc
                    out_dist['scale'] = scale
                    out_dist['arg'] = arg

            if verbose>=3:
                print("[DISTFIT] Checking for [%s] [SSE:%f]" %(dist.name,sse))

        except Exception:
            pass

    # Sort the output
    out = out.sort_values('SSE')
    out.reset_index(drop=True, inplace=True)
    # Return
    return(out, out_dist)


# Determine confidence intervals on the best fitting distribution
def _compute_cii(out_dist, alpha=None, bound='both'):
    # Separate parts of parameters
    arg = out_dist['params'][:-2]
    loc = out_dist['params'][-2]
    scale = out_dist['params'][-1]

    # Determine %CII
    dist = getattr(st, out_dist['name'])
    CIIup, CIIdown = None, None
    if alpha is not None:
        if bound=='up' or bound=='both' or bound=='right':
            CIIdown = dist.ppf(1 - alpha, *arg, loc=loc, scale=scale) if arg else dist.ppf(1 - alpha, loc=loc, scale=scale)
        if bound=='low' or bound=='both' or bound=='left':
            CIIup = dist.ppf(alpha, *arg, loc=loc, scale=scale) if arg else dist.ppf(alpha, loc=loc, scale=scale)

    # Store
    out_dist['CII_min_alpha']=CIIup
    out_dist['CII_max_alpha']=CIIdown
    return(out_dist)
