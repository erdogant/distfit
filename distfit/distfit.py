"""Compute best fit to your empirical distribution for 89 different theoretical distributions using the Residual Sum of Squares (RSS) estimates."""
# --------------------------------------------------
# Name        : distfit.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/distfit
# Licence     : MIT
# --------------------------------------------------


# %% Libraries
import pypickle
from distfit.utils.smoothline import smoothline

import warnings
warnings.filterwarnings('ignore')
import scipy.stats as st

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as multitest
import matplotlib.pyplot as plt


# %% Class dist
class distfit():
    """Probability density function fitting across 89 univariate distributions to non-censored data by residual sum of squares (RSS), making plots, and hypothesis testing.

    Description
    -----------
    Probability density fitting across 89 univariate distributions to
    non-censored data by Residual Sum of Squares (RSS), and hypothesis testing.

    Example
    -------
    >>> from distfit import distfit
    >>>
    >>> # Create dataset
    >>> X = np.random.normal(0, 2, 1000)
    >>> y = [-8,-6,0,1,2,3,4,5,6]
    >>>
    >>> # Set parameters
    >>> # Default method is set to parameteric models
    >>> dist = distfit()
    >>> # In case of quantile
    >>> dist = distfit(method='quantile')
    >>> # In case of quantile
    >>> dist = distfit(method='percentile')
    >>> # Fit using method
    >>> model_results = dist.fit_transform(X)
    >>> dist.plot()
    >>>
    >>> # Make prediction
    >>> results = dist.predict(y)
    >>> dist.plot()

    Parameters
    ----------
    method : str, default: 'parametric'
        Specify the method type: 'parametric','quantile','percentile'
    alpha : float, default: 0.05
        Significance alpha.
    multtest : str, default: 'fdr_bh'
        None, 'bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg',
        'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'
    bins : int, default: 50
        Bin size to determine the empirical historgram.
    bound : str, default: 'both'
        Set the directionality to test for significance.
        Upperbounds = 'up', 'high' or 'right', whereas lowerbounds = 'down', 'low' or 'left'
    distr : str, default: 'popular'
        The (set) of distribution to test. A set of distributions can be tested by:
        'popular', 'full', or specify the theoretical distribution: 'norm', 't'.
        See docs for more information about 'popular' and 'full'.
    smooth : int, default: None
        Smoothing the histogram can help to get a better fit when there are only few samples available.
    n_perm : int, default: 10000
        Number of permutations to model null-distribution in case of method is "quantile"

    Returns
    -------
    object.

    method : str
        Specified method for fitting and predicting.
    alpha : float
        Specified cut-off for P-value significance.
    bins : int
        Number of bins specified to create histogram.
    bound : str
        Specified testing directionality of the distribution.
    distr : str
        Specified distribution or a set of distributions.
    multtest : str
        Specified multiple test correction method.

    """

    def __init__(self, method='parametric', alpha=0.05, multtest='fdr_bh', bins=50, bound='both', distr='popular', smooth=None, n_perm=10000):
        """Initialize distfit with user-defined parameters."""
        if (alpha is None): alpha=1
        self.method = method
        self.alpha = alpha
        self.bins = bins
        self.bound = bound
        self.distr = distr
        self.multtest = multtest
        self.smooth = smooth
        self.n_perm = n_perm

    # Fit
    def fit(self, verbose=3):
        """Collect the required distribution functions.

        Parameters
        ----------
        verbose : int [1-5], default: 3
            Print information to screen. A higher number will print more.

        Returns
        -------
        Object.
        self.distributions : functions
            list of functions containing distributions.

        """
        if verbose>=3: print('[distfit] >fit..')
        # Get the desired distributions.
        if self.method=='parametric':
            self.distributions = _get_distributions(self.distr)
        elif self.method=='discrete':
            pass
            # TODO: https://stackoverflow.com/questions/62360325/determining-best-fit-distributions-by-sse-python-3-8/62489856#62489856
        elif self.method=='quantile':
            pass
        elif self.method=='percentile':
            pass
        else:
            raise Exception('[distfit] Error: method parameter can only be "parametric", "quantile" or "percentile".')

    # Transform
    def transform(self, X, verbose=3):
        """Determine best model for input data X.

        Description
        -----------
        The input data X can be modellend in two manners:

        **parametric**
            In the parametric case, the best fit on the data is determined using the
            Residual Sum of Squares approach (RSS) for the specified distributions. Based on
            the best distribution-fit, the confidence intervals (CII) can be determined
            for later usage in the :func:`predict` function.
        **quantile**
            In the quantile case, the data is ranked and the top/lower quantiles are determined.

        Parameters
        ----------
        X : array-like
            The Null distribution or background data is build from X.
        verbose : int [1-5], default: 3
            Print information to screen. A higher number will print more.

        Returns
        -------
        Object.
        model : dict
            dict containing keys with distribution parameters
            RSS : Residual Sum of Squares
            name : distribution name
            distr : distribution function
            params : all kind of parameters
            loc : loc function parameter
            scale : scale function parameter
            arg : arg function parameter
        summary : list
            Residual Sum of Squares
        histdata : tuple (observed, bins)
            tuple containing observed and bins for data X in the histogram.
        size : int
            total number of elements in for data X

        """
        if len(X)<1: raise Exception('[distfit] >Error: Input X is empty!')
        if verbose>=3: print('[distfit] >transform..')
        # Format the X
        X = _format_data(X)
        self.size = len(X)

        # Get histogram of original X
        [X_bins, y_obs] = _get_hist_params(X, self.bins)
        # Smoothing by interpolation
        [X_bins, y_obs] = smoothline(X_bins, y_obs, window=self.smooth, interpol=1, verbose=verbose)
        self.histdata = (y_obs, X_bins)

        if self.method=='parametric':
            # Compute best distribution fit on the empirical X
            out_summary, model = _compute_score_distribution(X, X_bins, y_obs, self.distributions, verbose=verbose)
            # Determine confidence intervals on the best fitting distribution
            model = _compute_cii(self, model, verbose=verbose)
            # Store
            self.model = model
            self.summary = out_summary
        elif self.method=='discrete':
            pass
        elif self.method=='quantile':
            # Determine confidence intervals on the best fitting distribution
            self.model = _compute_cii(self, X, verbose=verbose)
            self.summary = None
        elif self.method=='percentile':
            # Determine confidence intervals on the best fitting distribution
            self.model = _compute_cii(self, X, verbose=verbose)
            self.percentile = np.percentile(X, 7)
            self.summary = None
        else:
            raise Exception('[distfit] Error: method parameter can only be "parametric", "quantile" or "percentile".')

    # Fit and transform in one go
    def fit_transform(self, X, verbose=3):
        """Fit best scoring theoretical distribution to the empirical data (X).

        Parameters
        ----------
        X : array-like
            Set of values belonging to the data
        verbose : int [1-5], default: 3
            Print information to screen. A higher number will print more.

        Returns
        -------
        dict.
        model : dict
            dict containing keys with distribution parameters
            RSS : Residual Sum of Squares
            name : distribution name
            distr : distribution function
            params : all kind of parameters
            loc : loc function parameter
            scale : scale function parameter
            arg : arg function parameter
        summary : list
            Residual Sum of Squares
        histdata : tuple (observed, bins)
            tuple containing observed and bins for data X in the histogram.
        size : int
            total number of elements in for data X
            

        """
        # Fit model to get list of distributions to check
        self.fit(verbose=verbose)
        # Transform X based on functions
        self.transform(X, verbose=verbose)
        # Store
        results = _store(self.alpha,
                         self.bins,
                         self.bound,
                         self.distr,
                         self.histdata,
                         self.method,
                         self.model,
                         self.multtest,
                         self.n_perm,
                         self.size,
                         self.smooth,
                         self.summary,
                         )
        # Return
        return results


    def predict(self, y, verbose=3):
        """Compute probability for response variables y, using the specified method.

        Description
        -----------
        Computes P-values for [y] based on the fitted distribution from X.
        The empirical distribution of X is used to estimate the loc/scale/arg parameters for a
        theoretical distribution in case method type is ``parametric``.

        Parameters
        ----------
        y : array-like
            Values to be predicted.
        model : dict, default : None
            The model created by the .fit() function.
        verbose : int [1-5], default: 3
            Print information to screen. A higher number will print more.

        Returns
        -------
        Object.
        y_pred : list of str
            prediction of bounds [upper, lower] for input y, using the fitted distribution X.
        y_proba : list of float
            probability for response variable y.
        df : pd.DataFrame
            Dataframe containing the predictions in a structed manner.

        """
        if 'list' in str(type(y)): y=np.array(y)
        if 'float' in str(type(y)): y=np.array([y])
        if 'numpy.ndarray' not in str(type(y)): raise Exception('y should be of type np.array or list')
        if verbose>=3: print('[distfit] >predict..')

        if self.method=='parametric':
            out = _predict_parametric(self, y, verbose=verbose)
        elif self.method=='discrete':
            pass
        elif self.method=='quantile':
            out = _predict_quantile(self, y, verbose=verbose)
        elif self.method=='percentile':
            out = _predict_percentile(self, y, verbose=verbose)
        else:
            raise Exception('[distfit] Error: method parameter can only be "parametric", "quantile" or "percentile".')
        # Return
        return out

    # Plot
    def plot(self, title='', figsize=(10,8), xlim=None, ylim=None, verbose=3):
        """Make plot.

        Parameters
        ----------
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
        tuple (fig, ax)

        """
        if not hasattr(self, 'model'): raise Exception('[distfit] Error in plot: For plotting, A model is required. Try fitting first on your data using fit_transform(X)')
        if verbose>=3: print('[distfit] >plot..')
        if (self.method=='parametric'):
            fig, ax = _plot_parametric(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, verbose=verbose)
        elif (self.method=='discrete'):
            pass
        elif (self.method=='quantile'):
            fig, ax = _plot_quantile(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, verbose=verbose)
        elif (self.method=='percentile'):
            fig, ax = _plot_quantile(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, verbose=verbose)
        else:
            if verbose>=3: print('[distfit] >Warning: nothing to plot. Method not yet implemented for %s' %(self.method))
            fig, ax = None, None
        # Return
        return fig, ax

    # Plot summary
    def plot_summary(self, n_top=None, figsize=(15, 8), ylim=None, verbose=3):
        """Plot summary results.

        Parameters
        ----------
        n_top : int, optional
            Show the top number of results. The default is None.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        ylim : Float, optional (default: None)
            Limit figure in y-axis.
        verbose : Int [1-5], optional (default: 3)
            Print information to screen.

        Returns
        -------
        tuple (fig, ax)

        """
        if verbose>=3: print('[distfit] >plot summary..')
        if self.method=='parametric':
            if n_top is None:
                n_top = len(self.summary['RSS'])

            x = self.summary['RSS'][0:n_top]
            labels = self.summary['distr'].values[0:n_top]
            fig, ax = plt.subplots(figsize=figsize)
            plt.plot(x)
            # You can specify a rotation for the tick labels in degrees or with keywords.
            plt.xticks(np.arange(len(x)), labels, rotation='vertical')
            # Pad margins so that markers don't get clipped by the axes
            plt.margins(0.2)
            # Tweak spacing to prevent clipping of tick-labels
            plt.subplots_adjust(bottom=0.15)
            ax.grid(True)
            plt.xlabel('Distribution name')
            plt.ylabel('RSS (lower is better)')
            plt.title('Best fit: %s' %(self.model['name']))
            if ylim is not None:
                plt.ylim(ymin=ylim[0], ymax=ylim[1])

            plt.show()
            return(fig, ax)
        else:
            print('[distfit] This function works only in case of method is "parametric"')
            return None, None

    # Save model
    def save(self, filepath, verbose=3):
        """Save learned model in pickle file.

        Parameters
        ----------
        filepath : str
            Pathname to store pickle files.
        verbose : int, optional
            Show message. A higher number gives more informatie. The default is 3.

        Returns
        -------
        object

        """
        args = ['alpha','bins','bound','df','distr','distributions','histdata','method','model','multtest','n_perm','size','smooth','summary','y_pred']
        out = {}
        for arg in args:
            if hasattr(self, arg):
                if arg=='alpha': out.update({arg : self.alpha})
                if arg=='bins': out.update({arg : self.bins})
                if arg=='bound': out.update({arg : self.bound})
                if arg=='df': out.update({arg : self.df})
                if arg=='distr': out.update({arg : self.distr})
                if arg=='distributions': out.update({arg : self.distributions})
                if arg=='histdata': out.update({arg : self.histdata})
                if arg=='method': out.update({arg : self.method})
                if arg=='model': out.update({arg : self.model})
                if arg=='multtest': out.update({arg : self.multtest})
                if arg=='n_perm': out.update({arg : self.n_perm})
                if arg=='size': out.update({arg : self.size})
                if arg=='smooth': out.update({arg : self.smooth})
                if arg=='summary': out.update({arg : self.summary})
                if arg=='y_pred': out.update({arg : self.y_pred})

        status = pypickle.save(filepath, out, verbose=verbose)
        if verbose>=3: print('[distfit] >Saving.. %s' %(status))

    # Load model.
    def load(self, filepath, verbose=3):
        """Load learned model.

        Parameters
        ----------
        filepath : str
            Pathname to stored pickle files.
        verbose : int, optional
            Show message. A higher number gives more information. The default is 3.

        Returns
        -------
        Object.

        """
        out = pypickle.load(filepath, verbose=verbose)
        # Model Fit
        if out.get('model', None) is not None: self.model = out['model']
        if out.get('summary', None) is not None: self.summary = out['summary']
        if out.get('histdata', None) is not None: self.histdata = out['histdata']
        if out.get('size', None) is not None: self.size = out['size']
        # Parameters
        if out.get('smooth', None) is not None: self.smooth = out['smooth']
        if out.get('n_perm', None) is not None: self.n_perm = out['n_perm']
        if out.get('multtest', None) is not None: self.multtest = out['multtest']
        if out.get('method', None) is not None: self.method = out['method']
        if out.get('distributions', None) is not None: self.distributions = out['distributions']
        if out.get('distr', None) is not None: self.distr = out['distr']
        if out.get('bound', None) is not None: self.bound = out['bound']
        if out.get('bins', None) is not None: self.bins = out['bins']
        if out.get('alpha', None) is not None: self.alpha = out['alpha']
        # Predict
        if out.get('y_pred', None) is not None: self.y_pred = out['y_pred']
        if out.get('df', None) is not None: self.df = out['df']


# %%
def _predict_parametric(self, y, verbose=3):
    # Check which distribution fits best to the data
    if verbose>=4: print('[distfit] >Compute significance for y for the fitted theoretical distribution...')
    if not hasattr(self, 'model'): raise Exception('Error: Before making a prediction, a model must be fitted first using the function: fit_transform(X)')

    # Get distribution and the parameters
    getdist = self.model['distr']
    arg = self.model['params'][:-2]
    loc = self.model['params'][-2]
    scale = self.model['params'][-1]

    # Compute P-value for data based on null-distribution
    getP = getdist.cdf(y, *arg, loc, scale) if arg else getdist.pdf(y, loc, scale)

    # Determine P based on upper/lower/no bounds
    if self.bound=='up' or self.bound=='right' or self.bound=='high':
        Praw = 1 - getP
    elif self.bound=='down' or self.bound=='left' or self.bound=='low':
        Praw = getP
    elif self.bound=='both':
        Praw = np.min([1 - getP, getP], axis=0)
    else:
        raise Exception('[distfit] >Error in predict: "bounds" is not set correctly! Options are: up/down/right/left/high/low/both.')
        Praw=[]

    # Set all values in range[0..1]
    Praw = np.clip(Praw, 0, 1)
    # Multiple test correction
    y_proba = _do_multtest(Praw, self.multtest, verbose=verbose)
    # up/down based on threshold
    y_pred = np.repeat('none', len(y))
    if not isinstance(self.model['CII_max_alpha'], type(None)):
        if self.bound=='up' or self.bound=='right' or self.bound=='high' or self.bound=='both':
            y_pred[y>=self.model['CII_max_alpha']]='up'
    if not isinstance(self.model['CII_min_alpha'], type(None)):
        if self.bound=='down' or self.bound=='left' or self.bound=='low' or self.bound=='both':
            y_pred[y<=self.model['CII_min_alpha']]='down'

    # Make structured output
    df = pd.DataFrame()
    df['y'] = y
    df['y_proba'] = y_proba
    df['y_pred'] = y_pred
    df['P'] = Praw
    # Store in object
    self.df = df
    self.y_proba = y_proba
    self.y_pred = y_pred
    out = {}
    out['df'] = df
    out['y_proba'] = y_proba
    out['y_pred'] = y_pred
    return out


# %% _predict_quantile predict
def _predict_quantile(self, y, verbose=3):
    """Predict based on quantiles."""
    # Set bounds
    teststat = np.ones_like(y) * np.nan
    Praw = np.ones_like(y)

    # Predict
    y_pred = np.repeat('none', len(y))
    if not isinstance(self.model['CII_max_alpha'], type(None)):
        if self.bound=='up' or self.bound=='right' or self.bound=='high' or self.bound=='both':
            y_pred[y > self.model['CII_max_alpha']]='up'
    if not isinstance(self.model['CII_min_alpha'], type(None)):
        if self.bound=='down' or self.bound=='left' or self.bound=='low' or self.bound=='both':
            y_pred[y < self.model['CII_min_alpha']]='down'

    # Compute multiple testing to correct for Pvalues
    # y_proba = _do_multtest(Praw, self.multtest, verbose=verbose)
    Praw[np.isin(y_pred,['down','up'])] = 0

    # Make structured output
    df = pd.DataFrame()
    df['y'] = y
    df['y_proba'] = Praw
    df['y_pred'] = y_pred
    df['P'] = Praw
    df['teststat'] = teststat

    # Store in object
    self.df = df
    self.y_proba = Praw
    self.y_pred = y_pred

    out = {}
    out['df'] = df
    out['y_proba'] = Praw
    out['y_pred'] = y_pred
    return out


# %% percentile predict
def _predict_percentile(self, y, verbose=3):
    """Compute Probability based on quantiles.

    Description
    -----------
    Suppose you have 2 data sets with a unknown distribution and you want to test
    if some arbitrary statistic (e.g 7th percentile) is the same in the 2 data sets.
    An appropirate test statistic is the difference between the 7th percentile,
    and if we knew the null distribution of this statisic, we could test for the
    null hypothesis that the statistic = 0. Permuting the labels of the 2 data sets
    allows us to create the empirical null distribution.


    """
    # Set bounds
    teststat = np.ones_like(y) * np.nan
    Praw = np.ones_like(y)

    # Predict
    y_pred = np.repeat('none', len(y))
    if not isinstance(self.model['CII_max_alpha'], type(None)):
        if self.bound=='up' or self.bound=='right' or self.bound=='high' or self.bound=='both':
            y_pred[y > self.model['CII_max_alpha']]='up'
    if not isinstance(self.model['CII_min_alpha'], type(None)):
        if self.bound=='down' or self.bound=='left' or self.bound=='low' or self.bound=='both':
            y_pred[y < self.model['CII_min_alpha']]='down'

    # Compute statistics for y based on quantile distribution
    for i in range(0, len(y)):
        getstat = np.percentile(y[i], 7) - self.percentile
        # getP = (2 * np.sum(self.model['samples'] >= np.abs(getstat)) / self.n_perm)
        # getP = np.clip(getP, 0, 1)
        # Praw[i] = getP
        teststat[i] = getstat
        if verbose >= 4: print("[%.0f] - p-value = %f" %(y[i], getstat))

    Praw[np.isin(y_pred,['down','up'])]=0

    # Compute multiple testing to correct for Pvalues
    # y_proba = _do_multtest(Praw, self.multtest, verbose=verbose)
    y_proba = Praw

    # Make structured output
    df = pd.DataFrame()
    df['y'] = y
    df['y_proba'] = y_proba
    df['y_pred'] = y_pred
    df['P'] = Praw
    df['teststat'] = teststat
    # Store in object
    self.df = df
    self.y_proba = y_proba
    self.y_pred = y_pred
    out = {}
    out['df'] = df
    out['y_proba'] = y_proba
    out['y_pred'] = y_pred
    return out


# Plot
def _plot_quantile(self, title='', figsize=(15, 8), xlim=None, ylim=None, verbose=3):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot empirical data
    ax.plot(self.histdata[1], self.histdata[0], color='k', linewidth=1, label='empirical distribution')
    # add CII
    ax.axvline(self.model['CII_min_alpha'], linestyle='--', c='r', label='CII low')
    ax.axvline(self.model['CII_max_alpha'], linestyle='--', c='r', label='CII high')

    # Add significant hits as line into the plot. This data is dervived from dist.proba_parametric
    if hasattr(self, 'df'):
        for i in range(0, len(self.df['y'])):
            # if self.df['y_proba'].iloc[i]<=self.alpha and self.df['y_pred'].iloc[i] != 'none':
            if self.df['y_pred'].iloc[i] != 'none':
                ax.axvline(self.df['y'].iloc[i], c='g', linestyle='--', linewidth=0.8)

        idxIN = np.logical_or(self.df['y_pred']=='down', self.df['y_pred']=='up')
        if np.any(idxIN):
            ax.scatter(self.df['y'].values[idxIN], np.zeros(sum(idxIN)), color='g', marker='x', alpha=0.8, linewidth=1.5, label='Outside boundaries')
        idxOUT = self.df['y_pred']=='none'
        if np.any(idxOUT):
            ax.scatter(self.df['y'].values[idxOUT], np.zeros(sum(idxOUT)), color='r', marker='x', alpha=0.8, linewidth=1.5, label='Inside boundaries')

    # Limit axis
    if xlim is not None:
        plt.xlim(xmin=xlim[0], xmax=xlim[1])
    if ylim is not None:
        plt.ylim(ymin=ylim[0], ymax=ylim[1])

    ax.grid(True)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    plt.show()

    return fig, ax


# %% Plot
def _plot_parametric(self, title='', figsize=(10, 8), xlim=None, ylim=None, verbose=3):
    # Store output and function parameters
    model = self.model
    Param = {}
    Param['title'] = title
    Param['figsize'] = figsize
    Param['xlim'] = xlim
    Param['ylim'] = ylim

    # Make figure
    best_dist = model['distr']
    best_fit_name = model['name']
    best_fit_param = model['params']
    arg = model['params'][:-2]
    loc = model['params'][-2]
    scale = model['params'][-1]
    distline = getattr(st, model['name'])

    # Get pdf boundaries
    getmin = distline.ppf(0.0000001, *arg, loc=loc, scale=scale) if arg else distline.ppf(0.0000001, loc=loc, scale=scale)
    getmax = distline.ppf(0.9999999, *arg, loc=loc, scale=scale) if arg else distline.ppf(0.9999999, loc=loc, scale=scale)

    # Take maximum/minimum based on empirical data to avoid long theoretical distribution tails
    getmax = np.minimum(getmax, np.max(self.histdata[1]))
    getmin = np.maximum(getmin, np.min(self.histdata[1]))

    # Build pdf and turn into pandas Series
    x = np.linspace(getmin, getmax, self.size)
    y = distline.pdf(x, loc=loc, scale=scale, *arg)
    # ymax=max(self.histdata[0])

    fig, ax = plt.subplots(figsize=figsize)
    # Plot empirical data
    ax.plot(self.histdata[1], self.histdata[0], color='k', linewidth=1, label='empirical distribution')
    # Plot pdf
    ax.plot(x, y, 'b-', linewidth=1, label=best_fit_name)

    # Plot vertical line to stress the cut-off point
    if self.model['CII_min_alpha'] is not None:
        label = 'CII low ' + '(' + str(self.alpha) + ')'
        ax.axvline(x=model['CII_min_alpha'], ymin=0, ymax=1, linewidth=1.3, color='r', linestyle='dashed', label=label)
    if self.model['CII_max_alpha'] is not None:
        label = 'CII high ' + '(' + str(self.alpha) + ')'
        ax.axvline(x=model['CII_max_alpha'], ymin=0, ymax=1, linewidth=1.3, color='r', linestyle='dashed', label=label)

    # Make text for plot
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k, v in zip(param_names, best_fit_param)])
    ax.set_title('%s\n%s\n%s' %(Param['title'], best_fit_name, param_str))
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

    # Limit axis
    if Param['xlim'] is not None:
        plt.xlim(xmin=Param['xlim'][0], xmax=Param['xlim'][1])
    if Param['ylim'] is not None:
        plt.ylim(ymin=Param['ylim'][0], ymax=Param['ylim'][1])

    # Add significant hits as line into the plot. This data is dervived from dist.proba_parametric
    if hasattr(self, 'df'):
        # Plot significant hits
        if self.alpha is None: self.alpha=1

        idxIN=np.where(self.df['y_proba'].values<=self.alpha)[0]
        if verbose>=4: print("[distfit] >Plot Number of significant regions detected: %d" %(len(idxIN)))
        for i in idxIN:
            ax.axvline(x=self.df['y'].iloc[i], ymin=0, ymax=1, linewidth=1, color='g', linestyle='--', alpha=0.8)

        # Plot the samples that are not signifcant after multiple test.
        if np.any(idxIN):
            ax.scatter(self.df['y'].iloc[idxIN], np.zeros(len(idxIN)), color='g', marker='x', alpha=0.8, linewidth=1.5, label='Significant')

        # Plot the samples that are not signifcant after multiple test.
        idxOUT = np.where(self.df['y_proba'].values>self.alpha)[0]
        if np.any(idxOUT):
            ax.scatter(self.df['y'].values[idxOUT], np.zeros(len(idxOUT)), color='orange', marker='x', alpha=0.8, linewidth=1.5, label='Not significant')

    ax.legend()
    ax.grid(True)
    plt.show()

    if verbose>=4: print("[distfit] Estimated distribution: %s [loc:%f, scale:%f]" %(model['name'], model['params'][-2], model['params'][-1]))
    return (fig, ax)


# %% Utils
def _format_data(data):
    # Convert pandas to numpy
    if str(data.dtype)=='O': data=data.astype(float)
    if 'pandas' in str(type(data)): data = data.values
    # Make sure its a vector
    data = data.ravel()
    data = data.astype(float)
    return(data)


def _store(alpha, bins, bound, distr, histdata, method, model, multtest, n_perm, size, smooth, summary):
    out = {}
    out['model'] = model
    out['summary'] = summary
    out['histdata'] = histdata
    out['size'] = size
    out['alpha'] = alpha
    out['bins'] = bins
    out['bound'] = bound
    out['distr'] = distr
    out['method'] = method
    out['multtest'] = multtest
    out['n_perm'] = n_perm
    out['smooth'] = smooth
    # Return
    return out


# %% Get the distributions based on user input
def _get_distributions(distr):
    out_distr=[]

    # Get specified list of distributions
    if isinstance(distr, list):
        for getdistr in distr:
            try:
                out_distr.append(getattr(st, getdistr))
            except:
                print('[distfit] >Error: [%s] does not exist! <skipping>' %(getdistr))

    elif distr=='full':
        # st.levy_l, st.levy_stable
        out_distr = [st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
                         st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
                         st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
                         st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
                         st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
                         st.invweibull, st.johnsonsb, st.johnsonsu, st.laplace, st.levy,
                         st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami,
                         st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
                         st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda,
                         st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy]

    elif distr=='popular':
        out_distr = [st.norm, st.expon, st.pareto, st.dweibull, st.t, st.genextreme, st.gamma, st.lognorm, st.beta, st.uniform]
    else:
        # Collect distributions
        try:
            out_distr = [getattr(st, distr)]
        except:
            print('[distfit] >Error: [%s] does not exist! <skipping>' %(distr))
    
    if len(out_distr)==0: raise Exception('[distfit] >Error: Could nog select valid distributions for testing!')
    return(out_distr)


# %% Get histogram of original data
def _get_hist_params(X, bins, mhist='numpy'):
    if mhist=='numpy':
        [histvals, binedges] = np.histogram(X, bins=bins, density=True)
        binedges = (binedges + np.roll(binedges, -1))[:-1] / 2.0
        # binedges[-1] += 10**-6
    else:
        import seaborn as sns
        snsout = sns.distplot(X, bins=bins, norm_hist=False).get_lines()[0].get_data()
        histvals = snsout[1]
        binedges = snsout[0]
        binedges = np.append(binedges,10**-6)

    return(binedges, histvals)


# %% Compute score for each distribution
def _compute_score_distribution(data, X, y_obs, DISTRIBUTIONS, verbose=3):
    out = []
    model = {}
    model['distr'] = st.norm
    model['params'] = (0.0, 1.0)
    best_RSS = np.inf
    out = pd.DataFrame(index=range(0,len(DISTRIBUTIONS)), columns=['distr', 'RSS', 'LLE', 'loc', 'scale', 'arg'])
    max_name_len = np.max(list(map(lambda x: len(x.name), DISTRIBUTIONS)))

    # Estimate distribution parameters
    for i, distribution in enumerate(DISTRIBUTIONS):
        logLik = 0

        # Try to fit the dist. However this can result in an error so therefore you need to try-catch
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                # fit dist to data
                params = distribution.fit(data)
                if verbose>=5: print(params)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(X, loc=loc, scale=scale, *arg)
                # Compute RSS
                RSS = np.sum(np.power(y_obs - pdf, 2.0))
                logLik = np.nan

                # try:
                #     logLik = -np.sum( distribution.logpdf(y_obs, loc=loc, scale=scale) )
                # except Exception:
                #     pass
                # if len(params)>2:
                #     logLik = -np.sum( distribution.logpdf(y_obs, arg=arg, loc=loc, scale=scale) )
                # else:
                #     logLik = -np.sum( distribution.logpdf(y_obs, loc=loc, scale=scale) )

                # Store results
                out.values[i, 0] = distribution.name
                out.values[i, 1] = RSS
                out.values[i, 2] = logLik
                out.values[i, 3] = loc
                out.values[i, 4] = scale
                out.values[i, 5] = arg

                # identify if this distribution is better
                if best_RSS > RSS > 0:
                    best_RSS = RSS
                    model['name'] = distribution.name
                    model['distr'] = distribution
                    model['params'] = params
                    model['RSS'] = RSS
                    model['loc'] = loc
                    model['scale'] = scale
                    model['arg'] = arg

            if verbose>=3: print("[distfit] >[%s%s] [RSS: %.7f] [loc=%.3f scale=%.3f] " %(distribution.name, ' ' * (max_name_len - len(distribution.name)), RSS, loc, scale))

        except Exception:
            pass
            # e = sys.exc_info()[0]
            # if verbose>=1: print(e)

    # Sort the output
    out = out.sort_values('RSS')
    out.reset_index(drop=True, inplace=True)
    # Return
    return(out, model)


# %% Determine confidence intervals on the best fitting distribution
def _compute_cii(self, model, verbose=3):
    if verbose>=3: print("[distfit] >Compute confidence interval [%s]" %(self.method))
    CIIup, CIIdown = None, None
    if self.method=='parametric':
        # Separate parts of parameters
        arg = model['params'][:-2]
        loc = model['params'][-2]
        scale = model['params'][-1]

        # Determine %CII
        dist = getattr(st, model['name'])
        if self.alpha is not None:
            if self.bound=='up' or self.bound=='both' or self.bound=='right' or self.bound=='high':
                CIIdown = dist.ppf(1 - self.alpha, *arg, loc=loc, scale=scale) if arg else dist.ppf(1 - self.alpha, loc=loc, scale=scale)
            if self.bound=='down' or self.bound=='both' or self.bound=='left' or self.bound=='low':
                CIIup = dist.ppf(self.alpha, *arg, loc=loc, scale=scale) if arg else dist.ppf(self.alpha, loc=loc, scale=scale)
    elif self.method=='quantile':
        X = model
        model = {}
        CIIdown = np.quantile(X, 1 - self.alpha)
        CIIup = np.quantile(X, self.alpha)
        # model['model'] = model
    elif self.method=='percentile':
        X = model
        model = {}
        # Set Confidence intervals
        # ps = np.array([np.random.permutation(len(X)) for i in range(self.n_perm)])
        # xp = X[ps[:, :10]]
        # yp = X[ps[:, 10:]]
        # samples = np.percentile(xp, 7, axis=1) - np.percentile(yp, 7, axis=1)
        cii_high = (0 + (self.alpha / 2)) * 100
        cii_low = (1 - (self.alpha / 2)) * 100
        CIIup = np.percentile(X, cii_high)
        CIIdown = np.percentile(X, cii_low)
        # Store
        # model['samples'] = samples
    else:
        raise Exception('[distfit] Error: method parameter can only be "parametric", "quantile" or "percentile".')

    # Store
    model['CII_min_alpha'] = CIIup
    model['CII_max_alpha'] = CIIdown
    return(model)


# Multiple test correction
def _do_multtest(Praw, multtest='fdr_bh', verbose=3):
    """Multiple test correction for input pvalues.

    Parameters
    ----------
    Praw : list of float
        Pvalues.
    multtest : str, default: 'fdr_bh'
        Multiple testing method. Options are:
            None : No multiple testing
            'bonferroni' : one-step correction
            'sidak' : one-step correction
            'holm-sidak' : step down method using Sidak adjustments
            'holm' : step-down method using Bonferroni adjustments
            'simes-hochberg' : step-up method  (independent)
            'hommel' : closed method based on Simes tests (non-negative)
            'fdr_bh' : Benjamini/Hochberg  (non-negative)
            'fdr_by' : Benjamini/Yekutieli (negative)
            'fdr_tsbh' : two stage fdr correction (non-negative)
            'fdr_tsbky' : two stage fdr correction (non-negative)
    Returns
    -------
    list of float.
        Corrected pvalues.

    """
    if not isinstance(multtest, type(None)):
        if verbose>=3: print("[distfit] >Multiple test correction..[%s]" %multtest)
        if verbose>=5: print(Praw)
        Padj = multitest.multipletests(Praw, method=multtest)[1]
    else:
        Padj=Praw

    Padj = np.clip(Padj, 0, 1)
    return(Padj)
