"""Compute best fit to your empirical distribution for 89 different theoretical distributions using the Residual Sum of Squares (RSS) estimates."""
# --------------------------------------------------
# Name        : distfit.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/distfit
# Licence     : MIT
# --------------------------------------------------


# %% Libraries
import time
import pypickle
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from scipy.interpolate import make_interp_spline
import statsmodels.stats.multitest as multitest
import matplotlib.pyplot as plt
import scipy.stats as st

import warnings
warnings.filterwarnings('ignore')


# %% Class dist
class distfit():
    """Probability density function.

    Probability density fitting across 89 univariate distributions to non-censored data by scoring statistics
    such as residual sum of squares (RSS), making plots, and hypothesis testing.

    Parameters
    ----------
    method : str, default: 'parametric'
        Specify the method type.
            * 'parametric'
            * 'quantile'
            * 'percentile'
            * 'discrete'
    distr : str, default: 'popular'
        The (set) of distribution to test. A set of distributions can be tested by using a "popular" list or by specifying the theoretical distribution:
        In case using method="discrete", then binomial is used. See documentation for more information about 'popular' and 'full' (link reference below).
            * 'popular' : [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm, beta, uniform, st.oggamma]
            * 'full'
            * 'norm', 't', 'k': Test for one specific distribution.
            * ['norm', 't', 'k', ...]: Test for a list of distributions.
    stats : str, default: 'RSS'
        Specify the scoring statistics for the goodness of fit test.
            * 'RSS'
            * 'wasserstein'
            * 'ks': Kolmogorov-Smirnov statistic
            * 'energy'
    bins : int, default: 'auto'
        Bin size to determine the empirical historgram.
            * 'auto': Determine the bin size automatically.
            * 50: Set specific bin size
    bound : str, default: 'both'
        Set the directionality to test for significance.
            * 'up', 'high', 'right': Upperbounds
            * 'down', 'low' or 'left': lowerbounds
    alpha : float, default: 0.05
        Significance alpha.
    multtest : str, default: 'fdr_bh'
        Multiple test correction.
            * None
            * 'bonferroni'
            * 'sidak'
            * 'holm-sidak'
            * 'holm'
            * 'simes-hochberg'
            * 'hommel'
            * 'fdr_bh'
            * 'fdr_by'
            * 'fdr_tsbh'
            * 'fdr_tsbky'
    smooth : int, default: None
        Smoothing the histogram can help to get a better fit when there are only few samples available.
        The smooth parameter represnts a window that is used to create the convolution and gradually smoothen the line.
    n_perm : int, default: 10000
        Number of permutations to model null-distribution in case of method is "quantile"
    weighted : Bool, (default: True)
        Only used in discrete fitting, method="discrete". In principle, the best fit will be obtained if you set weighted=True. However, when using stats="RSS", you can set weighted=False.
    f : float, (default: 1.5)
        Only used in discrete fitting. It uses n in range n0/f to n0*f where n0 is the initial estimate.

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
    distr : str or list of strings
        Specified distribution or a set of distributions.
    multtest : str
        Specified multiple test correction method.
    todf : Bool (default: False)
        Output results in pandas dataframe when True. Note that creating pandas dataframes makes the code run significantly slower!

    Example
    -------
    >>> from distfit import distfit
    >>> import numpy as np
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

    References
    ----------
    * Documentation: https://erdogant.github.io/distfit/pages/html/Parametric.html

    """

    def __init__(self,
                 method='parametric',
                 distr: str = 'popular',
                 stats: str = 'RSS',
                 bins: int = 'auto',
                 bound: str = 'both',
                 alpha: float = 0.05,
                 multtest: str = 'fdr_bh',
                 smooth: int = None,
                 n_perm: int = 10000,
                 todf: bool = False,
                 weighted: bool = True,
                 f: float = 1.5,
                 mhist: str = 'numpy'
                 ):
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
        self.todf = todf
        self.stats = stats
        self.f = f  # Only for discrete
        self.weighted = weighted  # Only for discrete
        self.mhist = mhist

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
            self.distributions = self.get_distributions(self.distr)
        elif self.method=='discrete':
            # self.distributions = [st.binom]
            pass
        elif self.method=='quantile':
            pass
        elif self.method=='percentile':
            pass
        else:
            raise Exception('[distfit] Error: method parameter can only be "parametric", "discrete", "quantile" or "percentile".')

    # Transform
    def transform(self, X, verbose=3):
        """Determine best model for input data X.

        The input data X can be modellend in two manners:

        **parametric**
            In the parametric case, the best fit on the data is determined using the scoring statistic
            such as Residual Sum of Squares approach (RSS) for the specified distributions. Based on
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
            score : scoring statistic
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
        X_bins, y_obs = _get_hist_params(X, self.bins, mhist=self.mhist)
        # Smoothing by interpolation
        X_bins, y_obs = smoothline(X_bins, y_obs, interpol=1, window=self.smooth, verbose=verbose)
        self.histdata = (y_obs, X_bins)

        if self.method=='parametric':
            # Compute best distribution fit on the empirical X
            out_summary, model = _compute_score_distribution(X, X_bins, y_obs, self.distributions, self.stats, verbose=verbose)
            # Determine confidence intervals on the best fitting distribution
            model = _compute_cii(self, model, verbose=verbose)
            # Store
            self.model = model
            self.summary = out_summary
        elif self.method=='discrete':
            # Compute best distribution fit on the empirical X
            model, figdata = fit_transform_binom(X, f=self.f, weighted=True, stats=self.stats, verbose=verbose)
            model = _compute_cii(self, model, verbose=verbose)
            # self.histdata = (figdata['Xdata'], figdata['hist'])
            self.model = model
            self.summary = figdata
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
            score : Scoring statistic
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
        # Clean readily fitted models to ensure correct results.
        self._clean(verbose=verbose)
        # Fit model to get list of distributions to check
        self.fit(verbose=verbose)
        # Transform X based on functions
        self.transform(X, verbose=verbose)
        # Store
        results = _store(self.alpha,
                         self.stats,
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
                         self.weighted,
                         self.f,
                         )
        # Return
        return results

    def _clean(self, verbose=3):
        # Clean readily fitted models to ensure correct results.
        if hasattr(self, 'model'):
            if verbose>=3: print('[distfit] >Cleaning previous fitted model results..')
            if hasattr(self, 'histdata'): del self.histdata
            if hasattr(self, 'model'): del self.model
            if hasattr(self, 'summary'): del self.summary
            if hasattr(self, 'size'): del self.size

    def predict(self, y, alpha=None, verbose=3):
        """Compute probability for response variables y, using the specified method.

        Computes P-values for [y] based on the fitted distribution from X.
        The empirical distribution of X is used to estimate the loc/scale/arg parameters for a
        theoretical distribution in case method type is ``parametric``.

        Parameters
        ----------
        y : array-like
            Values to be predicted.
        model : dict, default : None
            The model created by the .fit() function.
        alpha : float, default: None
            Significance alpha is inherited from self if None.
        verbose : int [1-5], default: 3
            Print information to screen. A higher number will print more.

        Returns
        -------
        Object.
        y_pred : list of str
            prediction of bounds [upper, lower] for input y, using the fitted distribution X.
        y_proba : list of float
            probability for response variable y.
        df : pd.DataFrame (only when set: todf=True)
            Dataframe containing the predictions in a structed manner.

        """
        if 'list' in str(type(y)): y=np.array(y)
        if 'float' in str(type(y)): y=np.array([y])
        if 'numpy.ndarray' not in str(type(y)): raise Exception('y should be of type np.array or list')
        if alpha is not None:
            self.alpha = alpha
            if verbose>=3: print('[distfit] >Alpha is set to [%g]' %(self.alpha))

        if verbose>=3: print('[distfit] >predict..')

        if (self.method=='parametric') or (self.method=='discrete'):
        # if self.method=='parametric':
            out = _predict(self, y, verbose=verbose)
        # elif self.method=='discrete':
            # out = _predict(self, y, verbose=verbose)
        elif self.method=='quantile':
            out = _predict_quantile(self, y, verbose=verbose)
        elif self.method=='percentile':
            out = _predict_percentile(self, y, verbose=verbose)
        else:
            raise Exception('[distfit] >Error: method parameter can only be "parametric", "quantile" or "percentile".')
        # Return
        return out

    def generate(self, n, random_state=None, verbose=3):
        """Generate new samples based on the fitted distribution."""
        if not hasattr(self, 'model'): raise Exception('[distfit] Error in generate: A model is required. Try fitting first on your data using fit_transform(X)')
        if verbose>=3: print('[distfit] >Generate %s %s distributed samples with fitted params %s.' %(n, self.model['name'], str(self.model['params'])))
        X = None

        if (self.method=='parametric') or (self.method=='discrete'):
            X = self.model['model'].rvs(size=n, random_state=random_state)
        else:
            if verbose>=3: print('[distfit] >Warning: nothing to generate. Method should be of type: "parametric" or "discrete"')
        # Return
        return X

    # Plot
    def plot(self,
             title='',
             emp_properties={'color': '#000000', 'linewidth': 1.3, 'linestyle': '-', 'label': 'Emperical distribution'},
             pdf_properties={'color': '#004481', 'linewidth': 2, 'linestyle': '-'},
             bar_properties={'color': '#ffffff', 'linewidth': 1, 'edgecolor': '#808080', 'align': 'edge'},
             cii_properties={'color': '#880808', 'linewidth': 2, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r'},
             figsize=(20, 15),
             xlim=None,
             ylim=None,
             grid=True,
             fig=None,
             ax=None,
             verbose=3):
        """Make plot.

        Parameters
        ----------
        emp_properties : dict
            The line properties of the emperical line.
                * None: Do not plot.
                * {'color': '#000000', 'linewidth': 1.3, 'linestyle': '-', 'label': 'Emperical distribution'}: default
        pdf_properties : dict
            The line properties of the pdf.
                * None: Do not plot.
                * {'color': '#004481', 'linewidth': 2, 'linestyle': '-'}: default
        bar_properties : dict
            bar properties of the histogram.
                * None: Do not plot.
                * {'color': '#ffffff', 'linewidth': 1, 'edgecolor': '#808080', 'align': 'edge'}: default
        cii_properties : dict
            bar properties of the histogram.
                * None: Do not plot.
                * {'color': '#880808', 'linewidth': 2, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r'}: default
        title : String, optional (default: '')
            Title of the plot.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        xlim : Float, optional (default: None)
            Limit figure in x-axis.
        ylim : Float, optional (default: None)
            Limit figure in y-axis.
        grid : Bool, optional (default: True)
            Show the grid on the figure.
        fig : Figure, optional (default: None)
            Matplotlib figure (Note - ignored when method is `discrete`)
        ax : Axes, optional (default: None)
            Matplotlib Axes object (Note - ignored when method is `discrete`)
        verbose : Int [1-5], optional (default: 3)
            Print information to screen.

        Returns
        -------
        tuple (fig, ax)

        """
        if not hasattr(self, 'model'): raise Exception('[distfit] Error in plot: For plotting, A model is required. Try fitting first on your data using fit_transform(X)')
        if cii_properties is not None: cii_properties = {**{'color': '#880808', 'linewidth': 2, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r', 'alpha': 1}, **cii_properties}
        if emp_properties is not None: emp_properties = {**{'color': '#000000', 'linewidth': 1.3, 'linestyle': '-', 'label': 'Emperical distribution'}, **emp_properties}
        if pdf_properties is not None: pdf_properties = {**{'color': '#004481', 'linewidth': 2, 'linestyle': '-'}, **pdf_properties}
        if bar_properties is not None: bar_properties = {**{'color': '#ffffff', 'linewidth': 1, 'edgecolor': '#808080', 'align': 'edge'}, **bar_properties}

        if verbose>=3: print('[distfit] >plot..')
        if (self.method=='parametric'):
            fig, ax = _plot_parametric(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, fig=fig, ax=ax, grid=grid, emp_properties=emp_properties, pdf_properties=pdf_properties, bar_properties=bar_properties, cii_properties=cii_properties, verbose=verbose)
        elif (self.method=='discrete'):
            fig, ax = plot_binom(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, grid=grid, emp_properties=emp_properties, pdf_properties=pdf_properties, bar_properties=bar_properties, cii_properties=cii_properties, verbose=verbose)
        elif (self.method=='quantile') or (self.method=='percentile'):
            fig, ax = _plot_quantile(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, fig=fig, ax=ax, grid=grid, emp_properties=emp_properties, bar_properties=bar_properties, cii_properties=cii_properties, verbose=verbose)
        else:
            if verbose>=3: print('[distfit] >Warning: nothing to plot. Method not yet implemented for %s' %(self.method))
            fig, ax = None, None
        # Return
        return fig, ax

    # Plot summary
    def plot_summary(self, n_top=None, figsize=(15, 8), ylim=None, fig=None, ax=None, grid=True, verbose=3):
        """Plot summary results.

        Parameters
        ----------
        n_top : int, optional
            Show the top number of results. The default is None.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        ylim : Float, optional (default: None)
            Limit figure in y-axis.
        fig : Figure, optional (default: None)
            Matplotlib figure
        ax : Axes, optional (default: None)
            Matplotlib Axes object
        verbose : Int [1-5], optional (default: 3)
            Print information to screen.

        Returns
        -------
        tuple (fig, ax)

        """
        if verbose>=3: print('[distfit] >plot summary..')
        if self.method=='parametric':
            if n_top is None:
                n_top = len(self.summary['score'])

            x = self.summary['score'][0:n_top]
            labels = self.summary['distr'].values[0:n_top]
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

            plt.plot(x, color='#004481', linewidth=2)
            # You can specify a rotation for the tick labels in degrees or with keywords.
            plt.xticks(np.arange(len(x)), labels, rotation='vertical')
            # Pad margins so that markers don't get clipped by the axes
            plt.margins(0.2)
            # Tweak spacing to prevent clipping of tick-labels
            plt.subplots_adjust(bottom=0.15)
            ax.grid(grid)
            plt.xlabel('Distribution')
            plt.ylabel(('%s (lower is better)' %(self.stats)))
            plt.title('Best fit: %s' %(self.model['name']))
            if ylim is not None:
                plt.ylim(ymin=ylim[0], ymax=ylim[1])

            return (fig, ax)
        else:
            print('[distfit] This function works only in case of method is "parametric"')
            return None, None

    # Save model
    def save(self, filepath, overwrite=True, verbose=3):
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
        args = ['alpha', 'bins', 'bound', 'df', 'distr', 'distributions', 'histdata', 'method', 'model', 'multtest', 'n_perm', 'size', 'smooth', 'summary', 'y_pred', 'results']
        out = {}
        for arg in args:
            if hasattr(self, arg):
                if arg=='alpha': out.update({arg: self.alpha})
                if arg=='bins': out.update({arg: self.bins})
                if arg=='bound': out.update({arg: self.bound})
                if arg=='df': out.update({arg: self.df})  # TODO REMOVE
                if arg=='distr': out.update({arg: self.distr})
                if arg=='distributions': out.update({arg: self.distributions})
                if arg=='histdata': out.update({arg: self.histdata})
                if arg=='method': out.update({arg: self.method})
                if arg=='model': out.update({arg: self.model})
                if arg=='multtest': out.update({arg: self.multtest})
                if arg=='n_perm': out.update({arg: self.n_perm})
                if arg=='size': out.update({arg: self.size})
                if arg=='smooth': out.update({arg: self.smooth})
                if arg=='summary': out.update({arg: self.summary})
                if arg=='y_pred': out.update({arg: self.y_pred})
                if arg=='results': out.update({arg: self.results})

        status = pypickle.save(filepath, out, verbose=verbose, overwrite=overwrite)
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
        if out.get('results', None) is not None: self.results = out['results']

    # Get the distributions based on user input
    def get_distributions(self, distr='full'):
        """Return the distributions.

        Parameters
        ----------
        distr : str.
            Distributions to return.
                * 'full': all available distributions.
                * 'popular': Most common distributions.
                * 'norm', 't', 'k' or any other distribution name.
                * ['norm', 't', 'k']: list of distributions.

        Returns
        -------
        List with distributions.

        """
        out_distr=[]

        # Get specified list of distributions
        if isinstance(distr, list):
            for getdistr in distr:
                if getdistr=='k':
                    out_distr.append(k_distribution)
                else:
                    try:
                        out_distr.append(getattr(st, getdistr))
                    except:
                        print('[distfit] >Error: [%s] does not exist! <skipping>' %(getdistr))

        elif distr=='full':
            # st.levy_l, st.levy_stable, st.frechet_r, st.frechet_l
            out_distr = [st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
                             st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
                             st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
                             st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
                             st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
                             st.invweibull, st.johnsonsb, st.johnsonsu, st.laplace, st.levy,
                             st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami,
                             st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
                             st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda,
                             st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy]
    
        elif distr=='popular':
            out_distr = [st.norm, st.expon, st.pareto, st.dweibull, st.t, st.genextreme, st.gamma, st.lognorm, st.beta, st.uniform, st.loggamma]
        else:
            try:
                out_distr = [getattr(st, distr)]
            except:
                print('[distfit] >Error: [%s] does not exist! <skipping>' %(distr))

        if len(out_distr)==0: raise Exception('[distfit] >Error: Could nog select valid distributions for testing!')
        return out_distr


# %%
def _predict(self, y, verbose=3):
    # Check which distribution fits best to the data
    if verbose>=4: print('[distfit] >Compute significance for y for the fitted theoretical distribution...')
    if not hasattr(self, 'model'): raise Exception('Error: Before making a prediction, a model must be fitted first using the function: fit_transform(X)')

    # if (self.method=='parametric') or (self.method=='discrete'):
        # getdist = self.model['model']
    # Compute P-value for data based on null-distribution
    Pvalues = self.model['model'].cdf(y)
    # else:
    #     # Get distribution and the parameters
    #     getdist = self.model['distr']
    #     arg = self.model['params'][:-2]
    #     loc = self.model['params'][-2]
    #     scale = self.model['params'][-1]
    #     # Compute P-value for data based on null-distribution
    #     getP = getdist.cdf(y, *arg, loc, scale) if arg else getdist.cdf(y, loc, scale)

    # Determine P based on upper/lower/no bounds
    if self.bound=='up' or self.bound=='right' or self.bound=='high':
        Praw = 1 - Pvalues
    elif self.bound=='down' or self.bound=='left' or self.bound=='low':
        Praw = Pvalues
    elif self.bound=='both':
        Praw = np.min([1 - Pvalues, Pvalues], axis=0)
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
    self.y_proba = y_proba  # THIS WILL BE REMOVED IN NEWER VERSIONS
    self.y_pred = y_pred  # THIS WILL BE REMOVED IN NEWER VERSIONS
    self.y_bool = y_proba<=self.alpha
    self.results = {'y': y, 'y_proba': y_proba, 'y_pred': y_pred, 'P': Praw, 'y_bool': self.y_bool}
    if self.todf:
        # This approach is 3x faster then providing the dict to the dataframe
        self.df = pd.DataFrame(data=np.c_[y, y_proba, y_pred, Praw], columns=['y', 'y_proba', 'y_pred', 'P']).astype({'y': float, 'y_proba': float, 'y_pred': str, 'P': float})
        self.results['df'] = self.df

    # Return
    return self.results


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
    Praw[np.isin(y_pred, ['down', 'up'])] = 0

    # Store
    self.y_proba = Praw  # THIS WILL BE REMOVED IN NEWER VERSIONS
    self.y_pred = y_pred  # THIS WILL BE REMOVED IN NEWER VERSIONS
    self.results = {'y': y, 'y_proba': Praw, 'y_pred': y_pred, 'teststat': teststat}
    if self.todf:
        self.df = pd.DataFrame(data=np.c_[y, Praw, y_pred, Praw, teststat], columns=['y', 'y_proba', 'y_pred', 'P', 'teststat']).astype({'y': float, 'y_proba': float, 'y_pred': str, 'P': float, 'teststat': float})
        self.results['df'] = self.df

    # return
    return self.results


# %% percentile predict
def _predict_percentile(self, y, verbose=3):
    """Compute Probability based on quantiles.

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

    Praw[np.isin(y_pred, ['down', 'up'])] = 0

    # Compute multiple testing to correct for Pvalues
    # y_proba = _do_multtest(Praw, self.multtest, verbose=verbose)
    y_proba = Praw

    # Make structured output
    self.y_proba = y_proba  # THIS WILL BE REMOVED IN NEWER VERSIONS
    self.y_pred = y_pred  # THIS WILL BE REMOVED  IN NEWER VERSIONS
    self.results = {'y': y, 'y_proba': y_proba, 'y_pred': y_pred, 'P': Praw, 'teststat': teststat}
    if self.todf:
        self.df = pd.DataFrame(data=np.c_[y, y_proba, y_pred, Praw, teststat], columns=['y', 'y_proba', 'y_pred', 'P', 'teststat']).astype({'y': float, 'y_proba': float, 'y_pred': str, 'P': float, 'teststat': float})
        self.results['df'] = self.df

    # Return
    return self.results


# %% Plot bar
def _plot_bar(binedges, histvals, bar_properties, ax):
    if bar_properties is not None:
        ax.bar(binedges[:-1], histvals[:-1], width=np.diff(binedges), **bar_properties)


def _plot_pdf(x, y, label, pdf_properties, ax):
    if pdf_properties is not None:
        # Changing label of the pdf is not allowed.
        if pdf_properties.get('label', None) is not None: pdf_properties.pop('label')
        ax.plot(x, y, label=label, **pdf_properties)


def _plot_emp(x, y, line_properties, ax):
    if line_properties is not None:
        ax.plot(x, y, **line_properties)


def _get_cii_properties(cii_properties):
    cii_color = {}
    if cii_properties is not None:
        for p in ['color_sign_multipletest', 'color_sign', 'color_general', 'color', 'marker', 'size']:
            cii_color[p] = cii_properties.get(p)
            cii_properties.pop(p)
    cii_color = {**{'color': '#880808', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r'}, **cii_color}
    return cii_properties, cii_color


def _plot_cii_quantile(model, results, cii_properties, ax):
    if cii_properties is not None:
        # Extract cii properties
        cii_properties, cii_colors = _get_cii_properties(cii_properties)
        # add CII
        ax.axvline(model['CII_min_alpha'], c=cii_colors['color'], label='CII low', **cii_properties)
        ax.axvline(model['CII_max_alpha'], c=cii_colors['color'], label='CII high', **cii_properties)

        # Add significant hits as line into the plot. This data is dervived from dist.proba_parametric
        if results is not None:
            for i in range(0, len(results['y'])):
                if results['y_pred'][i] != 'none':
                    ax.axvline(results['y'][i], c=cii_colors['color_sign'], **cii_properties)

            idxIN = np.logical_or(results['y_pred']=='down', results['y_pred']=='up')
            if np.any(idxIN):
                cii_properties['label']='Outside boundaries'
                ax.scatter(results['y'][idxIN], np.zeros(sum(idxIN)), color=cii_colors['color_sign'], marker=cii_colors['marker'], **cii_properties)
            idxOUT = results['y_pred']=='none'
            if np.any(idxOUT):
                cii_properties['label']='Inside boundaries'
                ax.scatter(results['y'][idxOUT], np.zeros(sum(idxOUT)), color=cii_colors['color_general'], marker=cii_colors['marker'], **cii_properties)


# %% Plot
def _plot_quantile(self, title='', figsize=(15, 8), xlim=None, ylim=None, fig=None, ax=None, grid=True, emp_properties={}, bar_properties={}, cii_properties={}, verbose=3):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    if not hasattr(self, 'results'): self.results=None

    # Plot histogram empirical data
    _plot_bar(self.histdata[1], self.histdata[0], bar_properties, ax)
    # Plot empirical data
    _plot_emp(self.histdata[1], self.histdata[0], emp_properties, ax)
    # Plot CII
    _plot_cii_quantile(self.model, self.results, cii_properties, ax)

    # Limit axis
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid(grid)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend(loc='upper right')

    return fig, ax


# %% Plot
def _plot_parametric(self, title='', figsize=(10, 8), xlim=None, ylim=None, grid=True, fig=None, ax=None, emp_properties={}, pdf_properties={}, bar_properties={}, cii_properties={}, verbose=3):
    # Store output and function parameters
    model = self.model
    Param = {}
    Param['title'] = title
    Param['figsize'] = figsize
    Param['xlim'] = xlim
    Param['ylim'] = ylim
    cii_properties, cii_colors = _get_cii_properties(cii_properties)

    # Make figure
    best_dist = model['distr']
    best_fit_name = model['name'].title()
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

    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram empirical data
    _plot_bar(self.histdata[1], self.histdata[0], bar_properties, ax)
    # Plot empirical data
    _plot_emp(self.histdata[1], self.histdata[0], emp_properties, ax)
    # Plot pdf
    _plot_pdf(x, y, best_fit_name, pdf_properties, ax)

    # Make text for plot
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:g}'.format(k, v) for k, v in zip(param_names, best_fit_param)])
    ax.set_title('%s\n%s\n%s' %(Param['title'], best_fit_name, self.stats + ' (' + param_str + ')'))
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

    # Limit axis
    if Param['xlim'] is not None:
        ax.set_xlim(Param['xlim'][0], Param['xlim'][1])
    if Param['ylim'] is not None:
        ax.set_ylim(Param['ylim'][0], Param['ylim'][1])

    if cii_properties is not None:
        # Plot vertical line to stress the cut-off point
        if self.model['CII_min_alpha'] is not None:
            cii_properties['label'] = 'CII low ' + '(' + str(self.alpha) + ')'
            ax.axvline(x=model['CII_min_alpha'], ymin=0, ymax=1, color=cii_colors['color'], **cii_properties)
        if self.model['CII_max_alpha'] is not None:
            cii_properties['label'] = 'CII high ' + '(' + str(self.alpha) + ')'
            ax.axvline(x=model['CII_max_alpha'], ymin=0, ymax=1, color=cii_colors['color'], **cii_properties)
        if cii_properties.get('label'): cii_properties.pop('label')

    # Add significant hits as line into the plot. This data is dervived from dist.proba_parametric
    if hasattr(self, 'results') and (cii_properties is not None):
        if self.alpha is None: self.alpha=1
        idxIN=np.where(self.results['y_proba']<=self.alpha)[0]
        if verbose>=4: print("[distfit] >Plot Number of significant regions detected: %d" %(len(idxIN)))

        # Plot significant hits
        for i in idxIN:
            if cii_properties.get('label'): cii_properties.pop('label')
            ax.axvline(x=self.results['y'][i], ymin=0, ymax=1, markersize=cii_colors['size'], marker=cii_colors['marker'], color=cii_colors['color_sign_multipletest'], **cii_properties)

        # Plot the samples that are not signifcant after multiple test.
        if np.any(idxIN):
            cii_properties['label'] = 'Significant'
            ax.scatter(self.results['y'][idxIN], np.zeros(len(idxIN)), s=50, marker=cii_colors['marker'], color=cii_colors['color_sign'], **cii_properties)

        # Plot the samples that are not signifcant after multiple test.
        idxOUT = np.where(self.results['y_proba']>self.alpha)[0]
        if np.any(idxOUT):
            cii_properties['label'] = 'Not significant'
            ax.scatter(self.results['y'][idxOUT], np.zeros(len(idxOUT)), s=50, marker=cii_colors['marker'], color=cii_colors['color_general'], **cii_properties)

    ax.legend(loc='upper right')
    ax.grid(grid)

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
    return data


def _store(alpha, stats, bins, bound, distr, histdata, method, model, multtest, n_perm, size, smooth, summary, weighted, f):
    out = {}
    out['model'] = model
    out['summary'] = summary
    out['histdata'] = histdata
    out['size'] = size
    out['alpha'] = alpha
    out['stats'] = stats
    out['bins'] = bins
    out['bound'] = bound
    out['distr'] = distr
    out['method'] = method
    out['multtest'] = multtest
    out['n_perm'] = n_perm
    out['smooth'] = smooth
    out['weighted'] = weighted
    out['f'] = f
    # Return
    return out


# %% Get the distributions based on user input
def _get_distributions(distr):
    out_distr=[]

    # Get specified list of distributions
    if isinstance(distr, list):
        for getdistr in distr:
            if getdistr=='k':
                out_distr.append(k_distribution)
            else:
                try:
                    out_distr.append(getattr(st, getdistr))
                except:
                    print('[distfit] >Error: [%s] does not exist! <skipping>' %(getdistr))

    elif distr=='full':
        # st.levy_l, st.levy_stable, st.frechet_r, st.frechet_l
        out_distr = [st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
                         st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
                         st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
                         st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
                         st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
                         st.invweibull, st.johnsonsb, st.johnsonsu, st.laplace, st.levy,
                         st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami,
                         st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
                         st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda,
                         st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy]

    elif distr=='popular':
        out_distr = [st.norm, st.expon, st.pareto, st.dweibull, st.t, st.genextreme, st.gamma, st.lognorm, st.beta, st.uniform, st.loggamma]
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
        histvals, binedges = np.histogram(X, bins=bins, density=True)
        binedges = (binedges + np.roll(binedges, -1))[:-1] / 2.0
        # binedges[-1] += 10**-6
    else:
        import seaborn as sns
        snsout = sns.distplot(X, bins=bins, norm_hist=False).get_lines()[0].get_data()
        histvals = snsout[1]
        binedges = snsout[0]
        # binedges = np.append(binedges, 10**-6)

    return(binedges, histvals)


# %% Compute score for each distribution
def _compute_score_distribution(data, X, y_obs, DISTRIBUTIONS, stats, verbose=3):
    model = {}
    model['distr'] = st.norm
    model['stats'] = stats
    model['params'] = (0.0, 1.0)
    best_score = np.inf
    df = pd.DataFrame(index=range(0, len(DISTRIBUTIONS)), columns=['distr', 'score', 'LLE', 'loc', 'scale', 'arg'])
    # max_name_len = np.max(list(map(lambda x: len(x.name), DISTRIBUTIONS)))
    max_name_len = np.max(list(map(lambda x: len(x.name) if isinstance(x.name, str) else len(x.name()), DISTRIBUTIONS)))
    
    
    # Estimate distribution parameters
    for i, distribution in enumerate(DISTRIBUTIONS):
        logLik = np.nan

        # Fit the distribution. However this can result in an error so therefore you need to try-except
        try:
            start_time = time.time()

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
                # Compute score based on fit
                score = _compute_fit_score(stats, y_obs, pdf)
                # Get name of the distribution
                distr_name = distribution.name if isinstance(distribution.name, str) else distribution.name()

                # Store results
                # df.values[i, 0] = distribution.name
                df.values[i, 0] = distr_name
                df.values[i, 1] = score
                df.values[i, 2] = logLik
                df.values[i, 3] = loc
                df.values[i, 4] = scale
                df.values[i, 5] = arg

                # identify if this distribution is better
                if best_score > score > 0:
                    best_score = score
                    model['name'] = distr_name
                    model['distr'] = distribution
                    model['model'] = distribution(*arg, loc, scale) if arg else distribution(loc, scale)  # Store the fitted model
                    model['params'] = params
                    model['score'] = score
                    model['loc'] = loc
                    model['scale'] = scale
                    model['arg'] = arg

            if verbose>=3:
                spaces_1 = ' ' * (max_name_len - len(distr_name))
                scores = ('[%s: %g] [loc=%.3f scale=%.3f]' %(stats, score, loc, scale))
                time_spent = time.time() - start_time
                print("[distfit] >[%s%s] [%.4s sec] %s" %(distr_name, spaces_1, time_spent, scores))

        except Exception:
            pass
            # e = sys.exc_info()[0]
            # if verbose>=1: print(e)

    # Sort the output
    df = df.sort_values('score')
    df.reset_index(drop=True, inplace=True)
    # Return
    return(df, model)


# %% Compute fit score
def _compute_fit_score(stats, y_obs, pdf):
    if stats=='RSS':
        score = np.sum(np.power(y_obs - pdf, 2.0))
        # score = (((y_obs - pdf) / sigmas)**2).sum()
    elif stats=='wasserstein':
        score = st.wasserstein_distance(y_obs, pdf)
    elif stats=='energy':
        score = st.energy_distance(y_obs, pdf)
    elif stats=='ks':
        score = -np.log10(st.ks_2samp(y_obs, pdf)[1])
        # score = -np.log10(st.kstest(y_obs, pdf)[1])
    else:
        raise Exception('[%] statistic not implemented.', stats)
    return score


# %% Determine confidence intervals on the best fitting distribution
def _compute_cii(self, model, verbose=3):
    if verbose>=3: print("[distfit] >Compute confidence interval [%s]" %(self.method))
    CIIup, CIIdown = None, None

    if (self.method=='parametric') or (self.method=='discrete'):
        # Determine %CII
        if self.alpha is not None:
            if self.bound=='up' or self.bound=='both' or self.bound=='right' or self.bound=='high':
                # CIIdown = distr.ppf(1 - self.alpha, *arg, loc=loc, scale=scale) if arg else distr.ppf(1 - self.alpha, loc=loc, scale=scale)
                CIIdown = model['model'].ppf(1 - self.alpha)
            if self.bound=='down' or self.bound=='both' or self.bound=='left' or self.bound=='low':
                # CIIup = distr.ppf(self.alpha, *arg, loc=loc, scale=scale) if arg else distr.ppf(self.alpha, loc=loc, scale=scale)
                CIIup = model['model'].ppf(self.alpha)
    elif self.method=='quantile':
        X = model
        model = {}
        CIIdown = np.quantile(X, 1 - self.alpha)
        CIIup = np.quantile(X, self.alpha)
    elif self.method=='percentile':
        X = model
        model = {}
        # Set Confidence intervals
        cii_high = (0 + (self.alpha / 2)) * 100
        cii_low = (1 - (self.alpha / 2)) * 100
        CIIup = np.percentile(X, cii_high)
        CIIdown = np.percentile(X, cii_low)
    else:
        raise Exception('[distfit] >Error: method parameter can only be of type: "parametric", "quantile", "percentile" or "discrete".')

    # Store
    model['CII_min_alpha'] = CIIup
    model['CII_max_alpha'] = CIIdown
    return model


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
    return Padj


def smoothline(xs, ys=None, interpol=3, window=1, verbose=3):
    """Smoothing 1D vector.

    Smoothing a 1d vector can be challanging if the number of data is low sampled.
    This smoothing function therefore contains two steps. First interpolation of the
    input line followed by a convolution.

    Parameters
    ----------
    xs : array-like
        Data points for the x-axis.
    ys : array-like
        Data points for the y-axis.
    interpol : int, (default : 3)
        The interpolation factor. The data is interpolation by a factor n before the smoothing step.
    window : int, (default : 1)
        Smoothing window that is used to create the convolution and gradually smoothen the line.
    verbose : int [1-5], default: 3
        Print information to screen. A higher number will print more.

    Returns
    -------
    xnew : array-like
        Data points for the x-axis.
    ynew : array-like
        Data points for the y-axis.

    """
    if window is not None:
        if verbose>=3: print('[smoothline] >Smoothing by interpolation..')
        # Specify number of points to interpolate the data
        # Interpolate
        extpoints = np.linspace(0, len(xs), len(xs) * interpol)
        spl = make_interp_spline(range(0, len(xs)), xs, k=3)
        # Compute x-labels
        xnew = spl(extpoints)
        xnew[window:-window]

        # First smoothing on the raw input data
        ynew=None
        if ys is not None:
            ys = _smooth(ys, window)
            # Interpolate ys line
            spl = make_interp_spline(range(0, len(ys)), ys, k=3)
            ynew = spl(extpoints)
            ynew[window:-window]
    else:
        xnew, ynew = xs, ys
    return xnew, ynew


def _smooth(X, window):
    box = np.ones(window) / window
    X_smooth = np.convolve(X, box, mode='same')
    return X_smooth


# %% Binomial
class BinomPMF:
    """Wrapper so that integer parameters don't occur as function arguments.

    References
    ----------
    * Some parts of the binomial fitting is authored by Han-Kwang Nienhuys (2020); copying: CC-BY-SA.
    * https://stackoverflow.com/a/62365555/6228891
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, ks, p):
        """Compute binomial."""
        return st.binom(self.n, p).pmf(ks)


def transform_binom(hist, plot=True, weighted=True, f=1.5, stats='RSS', verbose=3):
    """Fit histogram to binomial distribution.

    Parameters
    ----------
    hist : array-like
        histogram as int array with counts, array index as bin.
    weighted : Bool, (default: True)
        In principle, the most best fit will be obtained if you set weighted=True.
        However, using different measures, such as minimum residual sum of squares (RSS) as a metric; you can set weighted=False.
    f : float, (default: 1.5)
        try to fit n in range n0/f to n0*f where n0 is the initial estimate.

    Returns
    -------
    model : dict
        distr : Object
            fitted binomial model.
        name : String
            Name of the fitted distribution.
        RSS : float
            Best RSS score
        n : int
            binomial n value.
        p : float
            binomial p value.
        chi2r : float
            rchi2: reduced chi-squared. This number should be around 1. Large values indicate a bad fit; small values indicate 'too good to be true' data..

    figdata : dict
        sses : array-like
            The computed RSS scores accompanyin the various n.
        Xdata : array-like
            Input data.
        hist : array-like
            fitted histogram as int array, same length as hist.
        Ydata : array-like
            Probability mass function.
        nvals : array-like
            Evaluated n's.
    """
    y_obs = hist / hist.sum()  # probability mass function
    nk = len(hist)
    if weighted:
        sigmas = np.sqrt(hist + 0.25) / hist.sum()
    else:
        sigmas = np.full(nk, 1 / np.sqrt(nk * hist.sum()))
    Xdata = np.arange(nk)

    mean = (y_obs * Xdata).sum()
    variance = ((Xdata - mean)**2 * y_obs).sum()

    # initial estimate for p and search range for n
    nest = max(1, int(mean**2 /(mean - variance) + 0.5))
    nmin = max(1, int(np.floor(nest / f)))
    nmax = max(nmin, int(np.ceil(nest * f)))
    nvals = np.arange(nmin, nmax + 1)
    num_n = nmax - nmin + 1
    if verbose>=4: print(f'[distfit] >Initial estimate: n={nest}, p={mean/nest:.3g}')

    # store fit results for each n
    pvals, scores = np.zeros(num_n), np.zeros(num_n)
    for nval in nvals:
        # Make quess for P
        p_guess = max(0, min(1, mean / nval))
        # Fit
        fitparams, _ = curve_fit(BinomPMF(nval), Xdata, y_obs, p0=p_guess, bounds=[0., 1.], sigma=sigmas, absolute_sigma=True)
        # Determine RSS
        p = fitparams[0]
        pdf = BinomPMF(nval)(Xdata, p)
        # Compute fit score
        score = _compute_fit_score(stats, y_obs, pdf)
        # Store
        pvals[nval - nmin] = p
        scores[nval - nmin] = score
        if verbose>=4: print('[distfit] >[binomial] [%s=%.3g] Trying n=%s -> p=%.3g, (initial=%.3g)' %(stats, score, nval, p, p_guess))

    n_fit = np.argmin(scores) + nmin
    p_fit = pvals[n_fit - nmin]
    score = scores[n_fit - nmin]
    chi2r = score / (nk - 2) if nk > 2 else np.nan
    if verbose>=3: print('[distfit] >[binomial] [%s=%.3g] [n=%.2g] [p=%.6g] [chi^2=%.3g]' %(stats, score, n_fit, p_fit, chi2r))

    # Store
    model = {}
    model['name'] = 'binom'
    model['distr'] = st.binom
    model['model'] = st.binom(n_fit, p_fit)
    model['params'] = (n_fit, p_fit)
    model['score'] = score
    model['chi2r'] = chi2r
    model['n'] = n_fit
    model['p'] = p_fit
    figdata = {}
    figdata['scores'] = scores
    figdata['Xdata'] = Xdata
    figdata['hist'] = hist
    figdata['Ydata'] = y_obs  # probability mass function
    figdata['nvals'] = nvals
    # Return
    return model, figdata


def fit_binom(X):
    """Transform array of samples (nonnegative ints) to histogram."""
    X = np.array(X, dtype=int)
    kmax = X.max()
    hist, _ = np.histogram(X, np.arange(kmax + 2) - 0.5)
    # force 1D int array
    hist = np.array(hist, dtype=int).ravel()
    return hist


def fit_transform_binom(X, f=1.5, weighted=True, stats='RSS', verbose=3):
    """Convert array of samples (nonnegative ints) to histogram and fit."""
    if verbose>=3: print('[distfit] >Fit using binomial distribution..')
    hist = fit_binom(X)
    model, figdata = transform_binom(hist, f=f, weighted=weighted, stats=stats, verbose=verbose)
    return model, figdata


def plot_binom(self,
               emp_properties={},
               pdf_properties={},
               bar_properties={},
               cii_properties={},
               title='',
               figsize=(10, 8),
               xlim=None,
               ylim=None,
               grid=True,
               verbose=3):
    """Plot discrete results.

    Parameters
    ----------
    model : dict
        Results derived from the fit_transform function.

    """
    # Store output and function parameters
    Param = {}
    Param['title'] = title
    Param['figsize'] = figsize
    Param['xlim'] = xlim
    Param['ylim'] = ylim
    # Make figure
    # dist = self.model['distr']
    best_fit_name = self.model['name'].title()
    best_fit_param = self.model['params']
    cii_properties, cii_colors = _get_cii_properties(cii_properties)

    model = self.model
    figdata = self.summary
    n_fit = model['n']
    p_fit = model['p']
    histf = BinomPMF(n_fit)(figdata['Xdata'], p_fit) * figdata['hist'].sum()

    # Init figure
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    # plot Emperical data
    if emp_properties is not None:
        bar_properties['align']='center'
        bar_properties['label']='Histogram'
        ax[0].bar(figdata['Xdata'], figdata['hist'], **bar_properties)
        ax[0].plot(figdata['Xdata'], figdata['hist'], 'o', color=emp_properties['color'], label=emp_properties['label'])

    # plot PDF
    if pdf_properties is not None:
        pdf_properties['label'] = 'PMF (binomial)'
        ax[0].step(figdata['Xdata'], histf, where='mid', **pdf_properties)
        ax[0].axhline(0, color=pdf_properties['color'])

    # Plot CII
    if cii_properties is not None:
        # Plot vertical line to stress the cut-off point
        if self.model['CII_min_alpha'] is not None:
            cii_properties['label'] = 'CII low ' + '(' + str(self.alpha) + ')'
            ax[0].axvline(x=model['CII_min_alpha'], ymin=0, ymax=1, color=cii_colors['color'], **cii_properties)
        if self.model['CII_max_alpha'] is not None:
            cii_properties['label'] = 'CII high ' + '(' + str(self.alpha) + ')'
            ax[0].axvline(x=model['CII_max_alpha'], ymin=0, ymax=1, color=cii_colors['color'], **cii_properties)

        # Add significant hits as line into the plot. This data is dervived from dist.proba_parametric
        if hasattr(self, 'results'):
            # Plot significant hits with multiple test
            if self.alpha is None: self.alpha=1
            idxIN=np.where(self.results['y_proba']<=self.alpha)[0]
            if verbose>=4: print("[distfit] >Plot Number of significant regions detected: %d" %(len(idxIN)))
            if cii_properties.get('label'): cii_properties.pop('label')
            for i in idxIN:
                ax[0].axvline(x=self.results['y'][i], ymin=0, ymax=1, markersize=cii_colors['size'], marker=cii_colors['marker'], color=cii_colors['color_sign_multipletest'], **cii_properties)

            # Plot the samples that signifcant without multiple test.
            if np.any(idxIN):
                cii_properties['label']='Significant'
                ax[0].scatter(self.results['y'][idxIN], np.zeros(len(idxIN)), size=cii_colors['size'], marker=cii_colors['marker'], color=cii_colors['color_sign'], **cii_properties)

            # Plot the samples that are not signifcant.
            idxOUT = np.where(self.results['y_proba']>self.alpha)[0]
            if np.any(idxOUT):
                cii_properties['label']='Not significant'
                ax[0].scatter(self.results['y'][idxOUT], np.zeros(len(idxOUT)), size=cii_colors['size'], marker=cii_colors['marker'], color=cii_colors['color_general'], **cii_properties)

    # Limit axis
    if Param['xlim'] is not None:
        ax[0].set_xlim(xmin=Param['xlim'][0], xmax=Param['xlim'][1])
    if Param['ylim'] is not None:
        ax[0].set_ylim(ymin=Param['ylim'][0], ymax=Param['ylim'][1])
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('Counts')
    ax[0].legend(loc='upper right')
    ax[0].grid(grid)
    param_str = ', '.join(['{}={:g}'.format(k, v) for k, v in zip(['n', 'p'], best_fit_param)])
    ax[0].set_title('%s\n%s\n%s' %(Param['title'], best_fit_name, param_str))
    ax[0].legend(loc='upper right')
    ax[0].grid(grid)

    # Second image
    ax[1].set_xlabel('n')
    ax[1].set_ylabel(self.stats)
    plotfunc = ax[1].semilogy if figdata['scores'].max()>20 * figdata['scores'].min()>0 else ax[1].plot
    plotfunc(figdata['nvals'], figdata['scores'], 'k-', label=('%s over n scan' %self.stats))
    ax[1].vlines(n_fit, 0, figdata['scores'].max(), color=cii_colors['color'], linestyles='dashed')
    ax[1].hlines(model['score'], figdata['nvals'].min(), figdata['nvals'].max(), color=cii_colors['color'], linestyles='dashed', label="Best %s: %.3g" %(self.stats, model['score']))
    ax[1].legend(loc='upper right')
    ax[1].grid(grid)
    fig.show()

    if verbose>=4: print("[distfit] Estimated distribution: %s [loc:%f, scale:%f]" %(model['name'], model['params'][-2], model['params'][-1]))
    return fig, ax


class k_distribution:
    """K-Distribution."""

    def __init__(self, loc=None, scale=None):
        self.loc=loc
        self.scale=scale

    def fit(X):
        """Fit for K-distribution.

        Parameters
        ----------
        X : Vector
            Numpy array containing data in vector form.

        Returns
        -------
        loc : Loc parameter
        scale : Scale parameter

        References
        ----------
        * 1. Rangaswamy M, Weiner D, Ozturk A. Computer generation of correlated non-Gaussian radar clutter[J]. IEEE Transactions on Aerospace and Electronic Systems, 1995, 31(1): 106-116.
        * 2. Lamont-Smith T. Translation to the normal distribution for radar clutter[J]. IEE Proceedings-Radar, Sonar and Navigation, 2000, 147(1): 17-22.
        * 3. https://en.wikipedia.org/wiki/K-distribution
        * 4. Redding N J. Estimating the parameters of the K distribution in the intensity domain[J]. 1999.

        """
        # K estimate
        x_2 = np.mean(X**2)
        x_4 = np.mean(X**4)
        scale = (x_4 / (2 * (x_2)**2) - 1)**(-1)
        loc = 0.5 * np.sqrt(x_2 / scale)
        return loc, scale

    def pdf(X, loc, scale):
        """Compute Probability Denity Distribution."""
        from scipy.special import gamma
        from scipy.special import kv as besselk
        f_k = 2 / (loc * gamma(scale)) * (X / (2 * loc))**scale * besselk(scale - 1, X / loc)
        return f_k

    def name():
        """Name of distribution."""
        return 'k'
