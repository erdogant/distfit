"""distfit is a python package for probability density fitting."""


# %% Libraries
import time
import pypickle
import numpy as np
import pandas as pd
from packaging import version
from joblib import Parallel, delayed

from scipy.optimize import curve_fit
import statsmodels.api as sm
# from tqdm import tqdm

from scipy.interpolate import make_interp_spline
import statsmodels.stats.multitest as multitest
import matplotlib.pyplot as plt
import scipy.stats as st
import logging
import colourmap

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('')
[logger.removeHandler(handler) for handler in logger.handlers[:]]
console = logging.StreamHandler()
formatter = logging.Formatter('[distfit] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
# logger.propagate = False


# %% Class dist
class distfit:
    """Probability density function.

    distfit is a python package for probability density fitting of univariate distributions for random variables.
    With the random variable as an input, distfit can find the best fit for parametric, non-parametric, and discrete distributions.

        * For the parametric approach, the distfit library can determine the best fit across 89 theoretical distributions.
          To score the fit, one of the scoring statistics for the good-of-fitness test can be used used, such as RSS/SSE, Wasserstein,
          Kolmogorov-Smirnov (KS), or Energy. After finding the best-fitted theoretical distribution, the loc, scale,
          and arg parameters are returned, such as mean and standard deviation for normal distribution.

        * For the non-parametric approach, the distfit library contains two methods, the quantile and percentile method.
          Both methods assume that the data does not follow a specific probability distribution. In the case of the quantile method,
          the quantiles of the data are modeled whereas for the percentile method, the percentiles are modeled.

        * In case the dataset contains discrete values, the distift library contains the option for discrete fitting.
          The best fit is then derived using the binomial distribution.

    Examples
    --------
    >>> from distfit import distfit
    >>> import numpy as np
    >>>
    >>> X = np.random.normal(0, 2, 1000)
    >>> y = [-8,-6,0,1,2,3,4,5,6]
    >>>
    >>> dfit = distfit()
    >>> results = dfit.fit_transform(X)
    >>>
    >>> # Plot summary
    >>> dfit.plot_summary()
    >>>
    >>> # PDF plot
    >>> dfit.plot()
    >>>
    >>> # Make prediction
    >>> results_proba = dfit.predict(y)
    >>>
    >>> # Plot PDF
    >>> fig, ax = dfit.plot(chart='pdf')
    >>>
    >>> # Add the CDF to the plot
    >>> fig, ax = dfit.plot(chart='cdf', n_top=1, ax=ax)
    >>>
    >>> # QQ-plot for top 10 fitted distributions
    >>> fig, ax = dfit.qqplot(X, n_top=10)
    >>>
    """

    def __init__(self,
                 method='parametric',
                 distr: str = 'popular',
                 stats: str = 'RSS',
                 bins: int = 'auto',
                 bound: str = 'both',
                 alpha: float = 0.05,
                 n_boots: int = None,
                 smooth: int = None,
                 n_perm: int = 10000,
                 todf: bool = False,
                 weighted: bool = True,
                 f: float = 1.5,
                 mhist: str = 'numpy',
                 cmap: str = 'Set1',
                 random_state: int = None,
                 verbose: [str, int] = 'info',
                 multtest=None,
                 n_jobs: int = 1,
                 ):
        """Initialize distfit with user-defined parameters.

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
                * 'popular' : [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm, beta, uniform, loggamma]
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
        n_boots : int, default: None
            Number of bootstraps to validate the fit.
                * None: No Bootstrap.
                * 1000: Thousand bootstraps.
        smooth : int, default: None
            Smoothing the histogram can help to get a better fit when there are only few samples available.
            The smooth parameter is a window that is used to create the convolution and gradually smoothen the line.
        n_perm : int, default: 10000
            Number of permutations to model null-distribution in case of method is "quantile"
        weighted : Bool, (default: True)
            Only used in discrete fitting, method="discrete". In principle, the best fit will be obtained if you set weighted=True. However, when using stats="RSS", you can set weighted=False.
        f : float, (default: 1.5)
            Only used in discrete fitting. It uses n in range n0/f to n0*f where n0 is the initial estimate.
        cmap : String, optional (default: 'Set1')
            Colormap when plotting multiple the CDF. The used colors are stored in dfit.summary['colors'].
        random_state : int, optional
            Random state.
        n_jobs : int, optional (default: 1)
            Number of cpu cores that are used for the computations.
            Note that the use of multiple cores occasionally causes a RuntimeWarning: invalid value encountered in log. The results can then be unriable. It is better to set n_jobs=1.
            -1: Use all cores
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

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
        todf : Bool (default: False)
            Output results in pandas dataframe when True. Note that creating pandas dataframes makes the code run significantly slower!

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>>
        >>> X = np.random.normal(0, 2, 1000)
        >>> y = [-8,-6,0,1,2,3,4,5,6]
        >>>
        >>> dfit = distfit()
        >>> results = dfit.fit_transform(X)
        >>>
        >>> # Plot summary
        >>> dfit.plot_summary()
        >>>
        >>> # PDF plot
        >>> dfit.plot()
        >>>
        >>> # Make prediction
        >>> results_proba = dfit.predict(y)
        >>>
        >>> # Plot PDF
        >>> fig, ax = dfit.plot(chart='pdf', n_top=1)
        >>>
        >>> # Add the CDF to the plot
        >>> fig, ax = dfit.plot(chart='cdf', n_top=1, ax=ax)
        >>>
        >>> # QQ-plot for top 10 fitted distributions
        >>> fig, ax = dfit.qqplot(X, n_top=10)
        >>>

        References
        ----------
            * https://erdogant.github.io/distfit

        """
        # Set the logger
        set_logger(verbose=verbose)

        if (alpha is None): alpha=1
        self.method = method
        self.alpha = alpha
        self.bins = bins
        self.bound = bound
        self.distr = distr
        self.n_boots = n_boots
        self.multtest = 'fdr_bh'
        self.smooth = smooth
        self.n_perm = n_perm
        self.todf = todf
        self.stats = stats
        self.f = f  # Only for discrete
        self.weighted = weighted  # Only for discrete
        self.mhist = mhist
        self.cmap = cmap
        self.random_state = random_state
        self.verbose = verbose

        # Set n_jobs
        if n_boots is not None:
            # Allocate the cores to bootstrapping
            self.n_jobs_dist = 1
            self.n_jobs = n_jobs
        else:
            # Allocate the cores to general proces
            self.n_jobs_dist = n_jobs
            self.n_jobs = 1

        if multtest is not None: logger.warning('multtest will be removed from initialization in a future release. Please set this parameter when using the predict function. Example: dfit.predict(multtest="holm")')
        # Check versions
        check_version()

    # Fit
    def fit(self, verbose=None):
        """Collect the required distribution functions.

        Returns
        -------
        Object.
        self.distributions : functions
            list of functions containing distributions.

        """
        if verbose is not None: set_logger(verbose)
        logger.info('fit')
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
            raise Exception('[distfit]> Error: method parameter can only be "parametric", "discrete", "quantile" or "percentile".')

    # Transform
    def transform(self, X, verbose=None):
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
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

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
        if verbose is not None: set_logger(verbose)
        if len(X)<1: raise Exception('[distfit] >Error: Input X is empty!')
        logger.info('transform')
        # Format the X
        X = _format_data(X)
        self.size = len(X)

        # Get histogram of original X
        X_bins, y_obs = self.density(X, self.bins, mhist=self.mhist)
        # Smoothing by interpolation
        X_bins, y_obs = smoothline(X_bins, y_obs, interpol=1, window=self.smooth)
        self.histdata = (y_obs, X_bins)

        if self.method=='parametric':
            # Compute best distribution fit on the empirical X
            out_summary, model = _compute_score_distribution(X, X_bins, y_obs, self.distributions, self.stats, cmap=self.cmap, n_boots=self.n_boots, random_state=self.random_state, n_jobs=self.n_jobs, n_jobs_dist=self.n_jobs_dist)
            # Determine confidence intervals on the best fitting distribution
            model = compute_cii(self, model, logger=logger)
            # Store
            self.model = model
            self.summary = out_summary
        elif self.method=='discrete':
            # Compute best distribution fit on the empirical X
            out_summary, model, figdata = fit_transform_binom(X, f=self.f, weighted=True, stats=self.stats)
            model = compute_cii(self, model, logger=logger)
            # self.histdata = (figdata['Xdata'], figdata['hist'])
            self.model = model
            self.summary = out_summary
            self.figdata = figdata
        elif self.method=='quantile':
            # Determine confidence intervals on the best fitting distribution
            self.model = compute_cii(self, X, logger=logger)
            self.summary = None
        elif self.method=='percentile':
            # Determine confidence intervals on the best fitting distribution
            self.model = compute_cii(self, X, logger=logger)
            self.percentile = np.percentile(X, 7)
            self.summary = None
        else:
            raise Exception(logger.error('Method parameter can only be "parametric", "quantile" or "percentile".'))

    # Fit and transform in one go.
    def fit_transform(self, X, n_boots=None, verbose=None):
        """Fit best scoring theoretical distribution to the empirical data (X).

        Parameters
        ----------
        X : array-like
            Set of values belonging to the data
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

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

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 1000)
        >>> y = [-8,-6,0,1,2,3,4,5,6]
        >>>
        >>> # Default method is parametric.
        >>> dfit = distfit()
        >>>
        >>> # In case of quantile
        >>> dfit = distfit(method='quantile')
        >>>
        >>> # In case of percentile
        >>> dfit = distfit(method='percentile')
        >>>
        >>> # Fit using method
        >>> model_results = dfit.fit_transform(X)
        >>>
        >>> dfit.plot()
        >>>
        >>> # Make prediction
        >>> results = dfit.predict(y)
        >>>
        >>> # Plot results with CII and predictions.
        >>> dfit.plot()
        >>>

        """
        if verbose is not None: set_logger(verbose)
        if n_boots is not None: self.n_boots=n_boots
        # Clean readily fitted models to ensure correct results.
        self._clean()
        # Fit model to get list of distributions to check
        self.fit()
        # Transform X based on functions
        self.transform(X)
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
                         self.n_boots,
                         self.random_state,
                         )
        # Return
        return results

    def _clean(self):
        # Clean readily fitted models to ensure correct results.
        if hasattr(self, 'model'):
            logger.info('Cleaning previous fitted model results.')
            if hasattr(self, 'histdata'): del self.histdata
            if hasattr(self, 'model'): del self.model
            if hasattr(self, 'summary'): del self.summary
            if hasattr(self, 'size'): del self.size

    def predict(self,
                y,
                alpha: float = None,
                multtest: str = 'fdr_bh',
                todf: bool = True,
                verbose: [str, int] = None):
        """Compute probability for response variables y, using the specified method.

        Computes P-values for [y] based on the fitted distribution from X.
        The empirical distribution of X is used to estimate the loc/scale/arg parameters for a
        theoretical distribution in case method type is ``parametric``.

        Parameters
        ----------
        y : array-like
            Values to be predicted.
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
        alpha : float, default: None
            Significance alpha is inherited from self if None.
        todf : Bool (default: False)
            Output results in pandas dataframe when True. Note that creating pandas dataframes makes the code run significantly slower!
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

        Returns
        -------
        Object.
        y_pred : list of str
            prediction of bounds [upper, lower] for input y, using the fitted distribution X.
        y_proba : list of float
            probability for response variable y.
        df : pd.DataFrame (only when set: todf=True)
            Dataframe containing the predictions in a structed manner.

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 1000)
        >>> y = [-8,-6,0,1,2,3,4,5,6]
        >>>
        >>> # Initialize
        >>> dfit = distfit(todf=True)
        >>> # Fit
        >>> model_results = dfit.fit_transform(X)
        >>>
        >>> # Make predictions
        >>> results = dfit.predict(y)
        >>> print(results['df'])
        >>>
        >>> # Plot results with CII and predictions.
        >>> dfit.plot()
        >>>
        """
        if verbose is not None: set_logger(verbose)
        if todf is not None: self.todf = todf
        if 'list' in str(type(y)): y=np.array(y)
        if 'float' in str(type(y)): y=np.array([y])
        if 'numpy.ndarray' not in str(type(y)): raise Exception('y should be of type np.array or list')
        self.multtest = multtest
        if alpha is not None:
            self.alpha = alpha
            logger.info('Alpha is set to [%g]' %(self.alpha))
            # Determine confidence intervals on the best fitting distribution
            self.model = compute_cii(self, self.model)

        logger.info('Compute significance for %d samples.' %(len(y)))

        if (self.method=='parametric') or (self.method=='discrete'):
            out = _predict(self, y)
        elif self.method=='quantile':
            out = _predict_quantile(self, y)
        elif self.method=='percentile':
            out = _predict_percentile(self, y)
        else:
            raise Exception('[distfit] >Error: method parameter can only be "parametric", "quantile" or "percentile".')
        # Return
        return out

    def generate(self, n, random_state=None, verbose=None):
        """Generate synthetic data based on the fitted distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        random_state : int, optional
            Random state.
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

        Returns
        -------
        X : np.array
            Numpy array with generated data.

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 1000)
        >>> y = [-8,-6,0,1,2,3,4,5,6]
        >>>
        >>> # Initialize
        >>> dfit = distfit()
        >>> # Fit
        >>> dfit.fit_transform(X)
        >>>
        >>> # Create syntethic data using fitted distribution.
        >>> Xnew = dfit.generate(10)
        >>>
        """
        if verbose is not None: set_logger(verbose)
        if random_state is not None: self.random_state = random_state
        if not hasattr(self, 'model') and not isinstance(self.distr, str): raise Exception('[distfit] Error in creating Synthetic data: A fitted model or input parameter "distr" is required. Tip: First fit on your data using dfit.fit_transform(X) or specify one distribution.')
        X = None

        if hasattr(self, 'model'):
            if (self.method=='parametric') or (self.method=='discrete'):
                logger.info('Create Synthetic data for %s %s distributed samples with fitted params %s.' %(n, self.model['name'], str(self.model['params'])))
                model = self.model['model']
            else:
                logger.warning('Nothing to generate. Method should be of type: "parametric" or "discrete"')
        elif isinstance(self.distr, str):
            logger.info('Create Synthetic data for [%s] distribution where parameters needs to be estimated first from a distribution (default: Uniform).' %(self.distr))
            model = eval('st.' + self.distr)
            # Set default parameters for the distribution of interest by fitting it to a normal distribution.
            X = st.uniform.rvs(size=1000)
            model = self.fit_transform(X, verbose=0)['model']['model']

        X = model.rvs(size=n, random_state=self.random_state)

        # Return
        return X

    # Get histogram density and bins.
    def density(self, X, bins='auto', mhist='numpy'):
        """Compute density based on input data and number of bins.

        Parameters
        ----------
        X : array-like
            Set of values belonging to the data
        bins : int, default: 'auto'
            Bin size to determine the empirical historgram.
                * 'auto': Determine the bin size automatically.
                * 50: Set specific bin size
        mhist : str, (default: 'numpy')
            The density extraction method.
                * 'numpy'
                * 'seaborn'

        Returns
        -------
        binedges : array-like
            Array with the bin edges.
        histvals : array-like
            Array with the histogram density values.

        Examples
        --------
        >>> from distfit import distfit
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 1000)
        >>>
        >>> # Initialize
        >>> dfit = distfit()
        >>>
        >>> # Compute bins and density
        >>> bins, density = dfit.density(X)
        >>>
        >>> # Make plot
        >>> plt.figure(); plt.plot(bins, density)
        >>>

        """
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

        return (binedges, histvals)

    # Plot
    def plot(self,
             chart='pdf',
             n_top=1,
             title='',
             emp_properties={'color': '#000000', 'linewidth': 3, 'linestyle': '-'},
             pdf_properties={'color': '#880808', 'linewidth': 3, 'linestyle': '-'},
             bar_properties={'color': '#607B8B', 'linewidth': 1, 'edgecolor': '#5A5A5A', 'align': 'center'},
             cii_properties={'color': '#C41E3A', 'linewidth': 3, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r'},
             fontsize=16,
             xlabel='Values',
             ylabel='Frequency',
             figsize=(20, 15),
             xlim=None,
             ylim=None,
             fig=None,
             ax=None,
             grid=True,
             cmap=None,
             verbose=None):
        """Make plot.

        Parameters
        ----------
        chart: str, default: 'pdf'
            Chart to plot.
                * 'pdf': Probability density function.
                * 'cdf': Cumulative density function.
        n_top : int, optional
            Show the top number of results. The default is 1.
        title : String, optional (default: '')
            Title of the plot.
        emp_properties : dict
            The line properties of the emperical line.
                * None: Do not plot.
                * {'color': '#000000', 'linewidth': 3, 'linestyle': '-'}
        pdf_properties : dict
            The line properties of the PDF or the CDF.
                * None: Do not plot.
                * {'color': '#880808', 'linewidth': 3, 'linestyle': '-'}
        bar_properties : dict
            bar properties of the histogram.
                * None: Do not plot.
                * {'color': '#607B8B', 'linewidth': 1, 'edgecolor': '#5A5A5A', 'align': 'edge'}
        cii_properties : dict
            bar properties of the histogram.
                * None: Do not plot.
                * {'color': '#C41E3A', 'linewidth': 3, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r'}
        fontsize : int, (default: 18)
            Fontsize for the axis and ticks.
        xlabel : String, (default: 'value')
            Label for the x-axis.
        ylabel : String, (default: 'Frequency')
            Label for the y-axis.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        xlim : Float, optional (default: None)
            Limit figure in x-axis.
        ylim : Float, optional (default: None)
            Limit figure in y-axis.
        fig : Figure, optional (default: None)
            Matplotlib figure (Note - ignored when method is `discrete`)
        ax : Axes, optional (default: None)
            Matplotlib Axes object (Note - ignored when method is `discrete`)
        grid : Bool, optional (default: True)
            Show the grid on the figure.
        cmap : String, optional (default: None)
            Colormap when plotting multiple the CDF. The used colors are stored in dfit.summary['colors'].
            However, when cmap is set, the specified colormap is used.
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

        Returns
        -------
        tuple (fig, ax)

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 10000)
        >>> y = [-8,-6,0,1,2,3,4,5,6]
        >>>
        >>> # Initialize
        >>> dfit = distfit(alpha=0.01)
        >>> dfit.fit_transform(X)
        >>> dfit.predict(y)
        >>>
        >>> # Plot seperately
        >>> fig, ax = dfit.plot(chart='pdf')
        >>> fig, ax = dfit.plot(chart='cdf')
        >>>
        >>> # Change or remove properties of the chart.
        >>> dfit.plot(chart='pdf', pdf_properties={'color': 'r'}, cii_properties={'color': 'g'}, emp_properties=None, bar_properties=None)
        >>> dfit.plot(chart='cdf', pdf_properties={'color': 'r'}, cii_properties={'color': 'g'}, emp_properties=None, bar_properties=None)
        >>>
        >>> # Create subplot
        >>> fig, ax = plt.subplots(1,2, figsize=(25, 10))
        >>> dfit.plot(chart='pdf', ax=ax[0])
        >>> dfit.plot(chart='cdf', ax=ax[1])
        >>>
        >>> # Change or remove properties of the chart.
        >>> fig, ax = dfit.plot(chart='pdf', pdf_properties={'color': 'r', 'linewidth': 3}, cii_properties={'color': 'r', 'linewidth': 3}, bar_properties={'color': '#1e3f5a'})
        >>> dfit.plot(chart='cdf', n_top=10, pdf_properties={'color': 'r'}, cii_properties=None, bar_properties=None, ax=ax)

        """
        if verbose is not None: set_logger(verbose)
        if not hasattr(self, 'model'): raise Exception('[distfit] Error in plot: For plotting, A model is required. Try fitting first on your data using fit_transform(X)')
        properties = _get_properties(pdf_properties, emp_properties, bar_properties, cii_properties)

        logger.info('Create %s plot for the %s method.' %(chart, self.method))
        if chart.lower()=='pdf' and self.method=='parametric':
            fig, ax = _plot_parametric(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, fig=fig, ax=ax, grid=grid, emp_properties=properties['emp'], pdf_properties=properties['pdf'], bar_properties=properties['bar'], cii_properties=properties['cii'], n_top=n_top, cmap=cmap, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
        elif chart.lower()=='pdf' and self.method=='discrete':
            fig, ax = plot_binom(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, grid=grid, emp_properties=properties['emp'], pdf_properties=properties['pdf'], bar_properties=properties['bar'], cii_properties=properties['cii'], xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
        elif chart.lower()=='pdf' and (self.method=='quantile') or (self.method=='percentile'):
            fig, ax = _plot_quantile(self, title=title, figsize=figsize, xlim=xlim, ylim=ylim, fig=fig, ax=ax, grid=grid, emp_properties=properties['emp'], bar_properties=properties['bar'], cii_properties=properties['cii'], xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
        elif chart.lower()=='cdf' and (self.method=='parametric' or self.method=='discrete'):
            fig, ax = self.plot_cdf(n_top=n_top, title=title, figsize=figsize, xlim=xlim, ylim=ylim, fig=fig, ax=ax, grid=grid, emp_properties=properties['emp'], cdf_properties=properties['pdf'], cii_properties=properties['cii'], cmap=cmap, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
        else:
            logger.warning('Nothing to plot. %s not yet implemented or possible for the %s approach.' %(chart, self.method))
            fig, ax = None, None

        # Return
        return fig, ax

    # QQ plot
    def qqplot(self,
               X,
               line='45',
               n_top=1,
               title='QQ-plot',
               fontsize=16,
               figsize=(20, 15),
               xlim=None,
               ylim=None,
               fig=None,
               ax=None,
               grid=True,
               alpha=0.5,
               size=15,
               cmap=None,
               verbose=None):
        """Plot QQplot results.

        Parameters
        ----------
        X : array-like
            The Null distribution or background data is build from X.
        line : str, default: '45'
            Options for the reference line to which the data is compared.
                * '45' - 45-degree line
                * 's' - standardized line, the expected order statistics are scaled by the standard deviation of the given sample and have the mean added to them.
                * 'r' - A regression line is fit
                * 'q' - A line is fit through the quartiles.
                * 'None' - by default no reference line is added to the plot.
        n_top : int, optional
            Show the top number of results. The default is 1.
        title : String, optional (default: '')
            Title of the plot.
        fontsize : int, (default: 18)
            Fontsize for the axis and ticks.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        xlim : Float, optional (default: None)
            Limit figure in x-axis.
        ylim : Float, optional (default: None)
            Limit figure in y-axis.
        fig : Figure, optional (default: None)
            Matplotlib figure (Note - ignored when method is `discrete`)
        ax : AxesSubplot, optional (default: None)
            Matplotlib Axes object. If given, this subplot is used to plot in instead of a new figure being created.
        grid : Bool, optional (default: True)
            Show the grid on the figure.
        cmap : String, optional (default: None)
            Colormap when plotting multiple the CDF. The used colors are stored in dfit.summary['colors'].
            However, when cmap is set, the specified colormap is used.
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

        Returns
        -------
        tuple (fig, ax)

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 1000)
        >>>
        >>> # Initialize
        >>> dfit = distfit()
        >>>
        >>> # Fit
        >>> dfit.fit_transform(X)
        >>>
        >>> # Make qq-plot
        >>> dfit.qqplot(X)
        >>>
        >>> # Make qq-plot for top 10 best fitted models.
        >>> dfit.qqplot(X, n_top=10)
        >>>

        """
        n_top = np.minimum(self.summary.shape[0], n_top)
        if cmap is not None: self.summary['color'] = colourmap.generate(self.summary.shape[0], cmap=cmap, scheme='hex', verbose=0)
        markeredgewidth = 0.5

        # Q-Q plot of the quantiles of x versus the quantiles/ppf of a distribution.
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        # Plot n
        for i in range(0, n_top):
            sm.qqplot(X,
                      line=line,
                      dist=self.summary['model'].iloc[i],
                      fit=False,
                      ax=ax,
                      **{'alpha': alpha,
                         'markersize': size,
                         'markerfacecolor': self.summary['color'].iloc[i],
                         'markeredgewidth': markeredgewidth,
                         'markeredgecolor': '#000000',
                         'marker': '.',
                         'label': self.summary['name'].iloc[i]},
                      )

        # Plot again to get the points at top.
        ax.set_title(self._make_title(title), fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xlabel('Theoretical Quantiles', fontsize=fontsize)
        ax.set_ylabel('Sample Quantiles', fontsize=fontsize)
        ax.legend(loc='upper left')
        ax.grid(grid)
        return fig, ax

    # Data plot
    def lineplot(self,
                 X,
                 labels=None,
                 projection=True,
                 xlabel='x-axes',
                 ylabel='y-axes',
                 title='',
                 fontsize=16,
                 figsize=(25, 12),
                 xlim=None,
                 ylim=None,
                 fig=None,
                 ax=None,
                 grid=True,
                 cii_properties={'alpha': 0.7, 'linewidth': 1},
                 line_properties={'linestyle': '-', 'color': '#004481', 'marker': '.', 'linewidth': 1, 'markersize': 10},
                 verbose=None):
        """Plot data and CII and/or predictions.

        Parameters
        ----------
        X : array-like
            The Null distribution or background data is build from X. The x-axis are the index values, and the y-axis the corresponding values.
        labels : array-like
            Labels for the x-axes. Should be the same size as X.
        projection : bool (default: True)
            Projection of the distribution.
        xlabel : string (default: 'Values')
            Label of the x-axis.
        ylabel : string (default: 'Frequencies')
            Label of the y-axis.
        title : String, optional (default: '')
            Title of the plot.
        fontsize : int, (default: 18)
            Fontsize for the axis and ticks.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        xlim : tuple, optional (default: None)
            Limit figure in x-axis: [0, 100]
        ylim : tuple, optional (default: None)
            Limit figure in y-axis.
            Limit figure in x-axis: [0, 10]
        fig : Figure, optional (default: None)
            Matplotlib figure (Note - ignored when method is `discrete`)
        ax : AxesSubplot, optional (default: None)
            Matplotlib Axes object. If given, this subplot is used to plot in instead of a new figure being created.
        grid : Bool, optional (default: True)
            Show the grid on the figure.
        cii_properties : dict
            bar properties of the histogram.
                * None: Do not plot.
                * {'color': '#C41E3A', 'linewidth': 3, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r'}
        line_properties : dict
            Properties of the line. Set one or multiple properties.
                * {'linestyle': '-', 'color': '#004481', 'marker': '.', 'linewidth': 1, 'markersize': 10}
                * {'color': '#000000'}
                * {'color': '#000000', 'marker': ''}
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

        Returns
        -------
        tuple (fig, ax)

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 1000)
        >>>
        >>> # Initialize
        >>> dfit = distfit()
        >>>
        >>> # Fit
        >>> dfit.fit_transform(X)
        >>>
        >>> # Make line plot
        >>> dfit.lineplot(X)
        >>>
        >>> # Make line plot
        >>> dfit.predict([0, 1, 2, 3, 4, 5])
        >>> dfit.lineplot(X)

        """
        cii_properties = _get_properties(None, None, None, cii_properties)['cii']
        line_properties = {**{'linestyle': '-', 'color': '#004481', 'marker': '.', 'linewidth': 1, 'markersize': 10}, **line_properties}

        if isinstance(X, pd.DataFrame):
            logger.info('Dataframe detected. Labels are derived from the index and the data is flattened.')
            labels = X.index.values
            X = X.values.ravel()

        # Make data input checks
        if (labels is not None) and len(X)!=len(labels): raise Exception('Labels should be of the same size as X')
        if labels is None: labels = range(0, len(X))
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        # Create projection
        if projection and hasattr(self, 'model'):
            _plot_projection(self, X, labels, line_properties, ax)

        # Create lineplot
        plt.plot(labels, X, **line_properties)

        if hasattr(self, 'model'):
            CII_min_alpha = self.model['CII_min_alpha']
            CII_max_alpha = self.model['CII_max_alpha']

            # Plot prediction results
            if hasattr(self, 'results'):
                Iloc = self.results['y_proba']<=self.alpha
                minth = self.results['y'][np.logical_and(self.results['y_pred']=='down', Iloc)]
                maxth = self.results['y'][np.logical_and(self.results['y_pred']=='up', Iloc)]

                # Mark significant regions.
                if len(maxth)>0:
                    ax.fill_between(labels, 0, 1, where=X >= maxth.min(), color='green', alpha=0.5, transform=ax.get_xaxis_transform(), label='Significantly Up')
                if len(minth)>0:
                    ax.fill_between(labels, 0, 1, where=X <= minth.max(), color='#880808', alpha=0.5, transform=ax.get_xaxis_transform(), label='Significantly Down')

            # Plot CII lines
            if CII_min_alpha is not None:
                ax.axhline(CII_min_alpha, color=cii_properties['color'], lw=cii_properties['linewidth'], alpha=cii_properties['alpha'], label='CII uperbound (alpha=' + str(self.alpha) + ')')
            if CII_max_alpha is not None:
                ax.axhline(CII_max_alpha, color=cii_properties['color'], lw=cii_properties['linewidth'], alpha=cii_properties['alpha'], label='CII lowerbound (alpha=' + str(self.alpha) + ')')

            # Make title
            title = self._make_title(title)

        # Limit axis
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        # Set figure properties
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.grid(grid)
        ax.legend(loc='upper left')

        # Return
        return fig, ax

    def _make_title(self, title=''):
        shapes = eval('st.' + self.model['name']).shapes
        param_names = (shapes + ', loc, scale').split(', ') if shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:g}'.format(k, v) for k, v in zip(param_names, self.model['params'])])
        # title = '%s\n%s\n%s' %(title, self.stats, self.model['name'] + '(' + param_str + ')')
        title = '%s\n%s' %(title, self.model['name'] + '(' + param_str + ')')
        return title

    # Plot CDF
    def plot_cdf(self,
                 n_top=1,
                 title='',
                 figsize=(20, 15),
                 xlabel='Values',
                 ylabel='Frequency',
                 fontsize=16,
                 xlim=None,
                 ylim=None,
                 fig=None,
                 ax=None,
                 grid=True,
                 emp_properties={'color': '#000000', 'linewidth': 1.3, 'linestyle': '-'},
                 cdf_properties={'color': '#004481', 'linewidth': 2, 'linestyle': '-'},
                 cii_properties={'color': '#880808', 'linewidth': 2, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r'},
                 cmap=None,
                 verbose=None):
        """Plot CDF results.

        Parameters
        ----------
        n_top : int, optional
            Show the top number of results. The default is 1.
        title : String, optional (default: '')
            Title of the plot.
        xlabel : string (default: 'Values')
            Label of the x-axis.
        ylabel : string (default: 'Frequencies')
            Label of the y-axis.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        xlim : tuple, optional (default: None)
            Limit figure in x-axis: [0, 100]
        ylim : tuple, optional (default: None)
            Limit figure in y-axis.
            Limit figure in x-axis: [0, 10]
        fig : Figure, optional (default: None)
            Matplotlib figure (Note - ignored when method is `discrete`)
        ax : Axes, optional (default: None)
            Matplotlib Axes object (Note - ignored when method is `discrete`)
        grid : Bool, optional (default: True)
            Show the grid on the figure.
        emp_properties : dict
            The line properties of the emperical line.
                * None: Do not plot.
                * {'color': '#000000', 'linewidth': 1.3, 'linestyle': '-'}: default
        cdf_properties : dict
            The line properties of the pdf.
                * None: Do not plot.
                * {'color': '#004481', 'linewidth': 2, 'linestyle': '-'}: default
        cmap : String, optional (default: None)
            Colormap when plotting multiple the CDF. The used colors are stored in dfit.summary['colors'].
            However, when cmap is set, the specified colormap is used.
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

        Returns
        -------
        tuple (fig, ax)

        Examples
        --------
        >>> from distfit import distfit
        >>> import numpy as np
        >>>
        >>> # Create dataset
        >>> X = np.random.normal(0, 2, 1000)
        >>>
        >>> # Initialize
        >>> dfit = distfit()
        >>>
        >>> # Fit
        >>> dfit.fit_transform(X)
        >>>
        >>> # Make CDF plot
        >>> fig, ax = dfit.plot(chart='cdf')
        >>>
        >>> # Append the PDF plot
        >>> dfit.plot(chart='pdf', fig=fig, ax=ax)
        >>>
        >>> # Plot the CDF of the top 10 fitted distributions.
        >>> fig, ax = dfit.plot(chart='cdf', n_top=10)
        >>> # Append the PDF plot
        >>> dfit.plot(chart='pdf', n_top=10, fig=fig, ax=ax)
        >>>
        """
        logger.info('Ploting CDF')
        if verbose is not None: set_logger(verbose)
        if n_top is None: n_top = 1
        properties = _get_properties(cdf_properties, emp_properties, None, cii_properties)

        # Create figure
        if self.method=='parametric' or self.method=='discrete':
            # Create figure
            if ax is None: fig, ax = plt.subplots(figsize=figsize)

            # Plot Emperical CDF
            count, bins_count = self.histdata
            # finding the PDF of the histogram using count values
            pdf_emp = count / sum(count)
            # using numpy np.cumsum to calculate the CDF. We can also find using the PDF values by looping and adding
            cdf_emp = np.cumsum(pdf_emp)
            # plot
            if properties['emp'] is not None:
                properties['emp']['marker'] = 'o'
                if properties['emp'].get('label', None) is None: properties['emp']['label'] = 'Emperical CDF'
                ax.plot(bins_count, cdf_emp, **properties['emp'])

            # Plot Theoretical CDF
            getmax = np.max(self.histdata[1])
            getmin = np.min(self.histdata[1])
            # Build pdf and turn into pandas Series
            x = np.linspace(getmin, getmax, self.size)
            if properties['pdf'] is not None:
                if properties['pdf'].get('label', None) is None: properties['pdf']['label'] = self.model['name'] + " (best fit)"
                cdf = self.model['model'].cdf
                # Plot the best CDF
                ax.plot(x, cdf(x), **properties['pdf'])

                # Plot other CDFs
                if n_top>1:
                    n_top = np.minimum(self.summary.shape[0], n_top + 1)
                    if cmap is not None: self.summary['color'] = colourmap.generate(self.summary.shape[0], cmap=cmap, scheme='hex', verbose=0)
                    for i in range(1, n_top):
                        # Plot cdf
                        cdf = self.summary['model'].iloc[i].cdf
                        # Plot CDF for linearly scale samples between min-max range(x)
                        ax.plot(x, cdf(x), **{'label': self.summary['name'].iloc[i], 'linewidth': 1.5, 'linestyle': '--', 'color': self.summary['color'].iloc[i]})

            # plot CII
            results = self.results if hasattr(self, 'results') else None
            _plot_cii_parametric(self.model, self.alpha, results, properties['cii'], ax)

            # Limit axis
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])

            # Make text for plot
            ax.set_title(self._make_title(title), fontsize=fontsize)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.legend(loc='upper right')
            ax.grid(grid)
        else:
            logger.warning('This function works only in case of method is "parametric"')
            return None, None
        return fig, ax

    # Plot summary
    def plot_summary(self,
                     n_top=None,
                     color_axes_left='#0000FF',
                     color_axes_right='#FC6600',
                     title=None,
                     rotation=45,
                     fontsize=16,
                     grid=True,
                     ylim=[None, None],
                     figsize=(20, 10),
                     fig=None,
                     ax=None,
                     verbose=None):
        """Plot summary results.

        Parameters
        ----------
        n_top : int, optional
            Show the top number of results. The default is None.
        figsize : tuple, optional (default: (10,8))
            The figure size.
        color_axes_left : str, (default: '#0000FF')
            Hex color of goodness of fit axes (left axes).
        color_axes_right : str, (default: '#FC6600')
            Hex color of boostrap axes (right axes).
        title : String, optional (default: '')
            Title of the plot.
        grid : Bool, optional (default: True)
            Show the grid on the figure.
        fig : Figure, optional (default: None)
            Matplotlib figure
        ylim : Float, optional (default: [None, None])
            Limit figure in y-axis.
        ax : Axes, optional (default: None)
            Matplotlib Axes object
        verbose : [str, int], default is 'info' or 20
            Set the verbose messages using string or integer values.
                * 0, 60, None, 'silent', 'off', 'no']: No message.
                * 10, 'debug': Messages from debug level and higher.
                * 20, 'info': Messages from info level and higher.
                * 30, 'warning': Messages from warning level and higher.
                * 50, 'critical': Messages from critical level and higher.

        Returns
        -------
        tuple (fig, ax)

        """
        if verbose is not None: set_logger(verbose)
        if n_top is None: n_top = len(self.summary['score'])
        if n_top==0: n_top = None
        logger.info('Ploting Summary.')

        # Create figure
        if self.method=='parametric':
            # Collect scores/data
            df = self.summary.iloc[0:n_top, :].copy()
            xcoord = np.arange(df.shape[0])

            # Create plot
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

            if ylim[1] is None:
                if ylim[0] is None: ylim[0] = -0.1
                # Create left axes
                score = scale_data(df['score'])
                ax.plot(score, color=color_axes_left, linewidth=1, linestyle='--')
                ax.scatter(xcoord, score, color=color_axes_left)

                # Round to a specific number of decimal places
                yticks = list(np.linspace(start=np.min(df['score']), stop=np.max(df['score']), num=len(ax.get_yticks()) - 2))
                yticks = [0] + yticks
                yticks = np.round(yticks, decimals=4)
                ax.set_yticklabels(yticks, fontsize=fontsize)
            else:
                ax.scatter(xcoord, df['score'], color=color_axes_left)
                score = df['score']
                ax.plot(score, color=color_axes_left, linewidth=1, linestyle='--')

            # You can specify a rotation for the tick labels in degrees or with keywords.
            ax.set_xticks(xcoord, df['name'].values, rotation=rotation)

            # Pad margins so that markers don't get clipped by the axes
            if title is None: title='%s (best fit)' %(self.model['name'].title())
            ax.margins(0.2)
            plt.subplots_adjust(bottom=0.15)
            ax.grid(grid)
            ax.set_xlabel('Probability Density Function (PDF)', fontsize=fontsize)
            ax.set_ylabel(('%s (goodness of fit test)' %(self.stats)), fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.set_ylim(ymin=ylim[0], ymax=ylim[1])

            # Create right axes
            if df['bootstrap_pass'][0] is not None:
                logger.info('Bootstrap results are included..')
                colors = np.array(['green' if x else 'red' for x in df['bootstrap_pass']])

                # color left y-axis
                ax.scatter(xcoord, score, color=colors)
                # Create right y-axis
                ax2 = ax.twinx()
                ax2.scatter(xcoord, df['bootstrap_score'], color=colors)
                ax2.plot(df['bootstrap_score'], color='#ff7f00', linewidth=1, linestyle='--')
                ax2.set_ylabel('Bootstrap score (higher is better)', fontsize=fontsize)
                ax2.set_ylim(ymin=-0.01, ymax=1)
                ax2.tick_params(axis='both', which='major', labelsize=fontsize)

                # Add legend
                from matplotlib.lines import Line2D
                green_dot = Line2D([0], [0], marker='o', color='green', label='Passed the 95% CII KS-test', markersize=10)
                red_dot = Line2D([0], [0], marker='o', color='red', label='Not passed the 95% CII KS-test', markersize=10)
                ax2.legend(handles=[green_dot, red_dot], loc='upper left')

                # host.yaxis.label.set_color(p1.get_color())
                ax.yaxis.label.set_color(color_axes_left)
                ax.tick_params(axis='y', colors=color_axes_left)
                ax2.yaxis.label.set_color(color_axes_right)
                ax2.tick_params(axis='y', colors=color_axes_right)

            # Show the plot
            # plt.show()
            return (fig, ax)
        else:
            logger.info('This function can only be used when method="parametric"')
            return None, None

    # Save model
    def save(self, filepath, overwrite=True):
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
        args = ['alpha', 'bins', 'bound', 'df', 'name', 'distributions', 'histdata', 'method', 'model', 'multtest', 'n_perm', 'size', 'smooth', 'summary', 'y_pred', 'results']
        out = {}
        for arg in args:
            if hasattr(self, arg):
                if arg=='alpha': out.update({arg: self.alpha})
                if arg=='bins': out.update({arg: self.bins})
                if arg=='bound': out.update({arg: self.bound})
                if arg=='df': out.update({arg: self.df})  # TODO REMOVE
                if arg=='name': out.update({arg: self.distr})
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

        status = pypickle.save(filepath, out, overwrite=overwrite)
        logger.info('Saving %s' %(status))

    # Load model.
    def load(self, filepath):
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
        out = pypickle.load(filepath)
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
        if out.get('name', None) is not None: self.distr = out['name']
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
                * 'popular' : [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm, beta, uniform, loggamma]
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
                        logger.error('[%s] does not exist! <skipping>' %(getdistr))

        elif distr=='full':
            # st.levy_l, st.levy_stable, st.frechet_r, st.frechet_l
            out_distr = [st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy,
                         st.chi, st.chi2, st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm,
                         st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm,
                         st.genlogistic, st.genpareto, st.gennorm, st.genexpon, st.genextreme, st.gausshyper, st.gamma,
                         st.gengamma, st.genhalflogistic, st.gibrat, st.gompertz, st.gumbel_r, st.gumbel_l,
                         st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
                         st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.laplace, st.levy,
                         st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke,
                         st.nakagami, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm,
                         st.rdist, st.reciprocal, st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t,
                         st.triang, st.truncexpon, st.truncnorm, st.tukeylambda, st.uniform, st.vonmises,
                         st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy]

        elif distr=='popular':
            out_distr = [st.norm, st.expon, st.pareto, st.dweibull, st.t, st.genextreme, st.gamma, st.lognorm, st.beta, st.uniform, st.loggamma]
        else:
            try:
                out_distr = [getattr(st, distr)]
            except:
                logger.error('[%s] does not exist! <skipping>' %(distr))

        if len(out_distr)==0: raise Exception('[distfit] >Error: [%s] is not a valid selection for distributions. Choose: "full", "popular", "norm", "t" etc or a list of distributions.' %(distr))
        return out_distr

    # bootstrap.
    def bootstrap(self, X, n_boots=100, alpha=0.05, n=10000, n_top=None, update_model=True):
        """Bootstrap.

        To validate our fitted model, the Kolmogorov-Smirnov (KS) test is used to compare the distribution of
        the bootstrapped samples to the original data to assess the goodness of fit. If the model is overfitting,
        the KS test will reveal a significant difference between the bootstrapped samples and the original data,
        indicating that the model is not representative of the underlying distribution.

        The goal here is to estimate the KS statistic of the fitted distribution when the params are estimated from data.
            1. Resample using fitted distribution.
            2. Use the resampled data to fit the distribution.
            3. Compare the resampled data vs. fitted PDF.
            4. Repeat 1000 times the steps 1-3
            5. return score=ratio succes / n_boots
            6. return whether the 95% CII for the KS-test statistic is valid.

        Parameters
        ----------
        X : array-like
            Set of values belonging to the data
        n_boots : int, default: None
            Number of bootstraps to validate the fit.
                * None: No Bootstrap.
                * 1000: Thousand bootstraps.
        alpha : float, default: 0.05
            Significance alpha.
        n : int, default: 10000
            Number of samples to draw per bootstrap. This number if set to minimum(len(X), n)
        n_top : int, optional
            Show the top number of results. The default is None.
        update_model : float, default: True
            Update to the best model.

        Returns
        -------
        None.

        Examples
        --------
        >>> # Import library
        >>> from distfit import distfit
        >>>
        >>> # Initialize with 100 permutations
        >>> dfit = distfit(n_boots=100)
        >>>
        >>> # Random data
        >>> # X = np.random.exponential(0.5, 10000)
        >>> # X = np.random.uniform(0, 1000, 10000)
        >>> X = np.random.normal(163, 10, 10000)
        >>>
        >>> results = dfit.fit_transform(X)
        >>>
        >>> # Results are stored in summary
        >>> dfit.summary[['name', 'score', 'bootstrap_score', 'bootstrap_pass']]
        >>>
        >>> # Create summary plot
        >>> dfit.plot_summary()

        Examples
        --------
        >>> # Import library
        >>> from distfit import distfit
        >>>
        >>> # Initialize without permutations
        >>> dfit = distfit()
        >>>
        >>> # Random data
        >>> # X = np.random.exponential(0.5, 10000)
        >>> # X = np.random.uniform(0, 1000, 10000)
        >>> X = np.random.normal(163, 10, 10000)
        >>>
        >>> # Fit without permutations
        >>> results = dfit.fit_transform(X)
        >>>
        >>> # Results are stored in summary
        >>> dfit.summary[['name', 'score', 'bootstrap_score', 'bootstrap_pass']]
        >>>
        >>> # Create summary plot (no bootstrap is present)
        >>> dfit.plot_summary()
        >>>
        >>> results = dfit.bootstrap(X, n_boots=100)
        >>>
        >>> # Create summary plot (the bootstrap is automatically added to the plot)
        >>> dfit.plot_summary()

        """
        if update_model and n_boots<10:
            logger.warning('Bootstrapping requires n_boots to be >=10 <return>')
            return None

        if n_top is None: n_top = self.summary.shape[0]
        self.summary['bootstrap_score'] = 0
        self.summary['bootstrap_pass'] = None
        logger.info('Bootstrap for %d distributions with n_boots=%d' %(n_top, n_boots))
        max_name_len = np.max(list(map(len, self.summary['name'][0:n_top].values)))
        for i in range(n_top):
            distr = self.summary['name'].iloc[i]
            bootstrap_score, bootstrap_pass = _bootstrap(eval('st.' + distr), self.summary['model'].iloc[i], X, n_boots=n_boots, alpha=alpha, random_state=self.random_state, n_jobs=self.n_jobs)
            # Store results
            logger.info('Bootstrap: [%s%s] > Score: %.2g > Pass 95%% CII KS-test: %s' %(distr, ' ' * (max_name_len - len(distr)), bootstrap_score, bootstrap_pass))
            self.summary['bootstrap_score'].iloc[i] = bootstrap_score
            self.summary['bootstrap_pass'].iloc[i] = bootstrap_pass

        # Sort on best model
        df_summary, model = _sort_dataframe(self.summary, cmap=self.cmap)

        # Save results
        if update_model and (n_boots is not None) and (n_boots>=10):
            logger.info('Updating model to: [%s]' %(model['name'].title()))
            # Determine confidence intervals on the best fitting distribution
            self.model = compute_cii(self, model)
            self.summary = df_summary
        # Return
        return df_summary

    def import_example(self, data='gas_spot_price'):
        """Import example dataset from github source.

        Imports data directly from github source.

        Parameters
        ----------
        data : str
            * 'gas_spot_price'
            * 'tips'
            * 'occupancy'

        Returns
        -------
        pd.DataFrame
            DataFrame that conains the data.

        """
        df = None
        logger.info('Downloading and processing [%s] from github source.' %(data))

        if data=='gas_spot_price':
            df = pd.read_csv(r'https://erdogant.github.io/datasets/Henry_Hub_Natural_Gas_Spot_Price.zip')
            df['Day'] = pd.to_datetime(df['Day'])
            df.dropna(inplace=True)
            df = df.rename(columns={'Day': 'date', 'Henry Hub Natural Gas Spot Price Dollars per Million Btu': 'price'})
            # Set the date column as the index
            df.set_index('date', inplace=True)
        elif data=='tips':
            df = pd.read_csv(r'https://erdogant.github.io/datasets/tips.zip')
        elif data=='occupancy':
            df = pd.read_csv(r'https://erdogant.github.io/datasets/UCI_Occupancy_Detection.zip')
        else:
            logger.error('[%s] is not a valid data set that can be returned.' %(data))

        return df

# %% Plot projection
def _plot_projection(self, X, labels, line_properties, ax):
    # Add horizontal lines
    minvalue = labels[-1] + int(labels[-1]*0.1)
    maxvalue = labels[-1] + int(labels[-1]*0.2)

    # Get histogram of original X
    X_bins, y_obs = self.density(X, self.bins, mhist=self.mhist)
    # Smoothing by interpolation
    X_bins, y_obs = smoothline(X_bins, y_obs, interpol=1, window=self.smooth)
    histdata = (y_obs, X_bins)


    # Rescale data
    pdf_y=scale_data_minmax(histdata[0], minvalue=minvalue, maxvalue=maxvalue)
    pdf_x=scale_data_minmax(np.arange(0, len(pdf_y)), minvalue=min(X), maxvalue=max(X))

    # Create emperical PDF
    ax.plot(pdf_y, pdf_x, color='black', linestyle='-', linewidth=1.5, label='Emperical PDF.')

    # Create vertical lines
    ax.vlines(x=minvalue, ymin=min(pdf_x), ymax=max(pdf_x), color='#000000', linewidth=1.2)

    # Create the horizontal lines
    for i in range(len(labels)):
        ax.hlines(X[i], labels[i], minvalue, colors='gray', linestyles='--', linewidth=0.8)
        ax.plot(minvalue, X[i], marker=line_properties['marker'], color=line_properties['color'], alpha=0.8, markersize=line_properties['markersize'])

    # Create PDF
    # Make figure
    best_fit_name = self.model['name'].title()
    arg = self.model['params'][:-2]
    loc = self.model['params'][-2]
    scale = self.model['params'][-1]
    distribution = getattr(st, self.model['name'])
    # Get pdf boundaries
    getmin = distribution.ppf(0.0000001, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.0000001, loc=loc, scale=scale)
    getmax = distribution.ppf(0.9999999, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.9999999, loc=loc, scale=scale)
    # Take maximum/minimum based on empirical data to avoid long theoretical distribution tails
    getmax = np.minimum(getmax, np.max(self.histdata[1]))
    getmin = np.maximum(getmin, np.min(self.histdata[1]))
    # Build pdf and turn into pandas Series
    x = np.linspace(getmin, getmax, self.size)
    y = distribution.pdf(x, loc=loc, scale=scale, *arg)
    y = scale_data_minmax(y, minvalue=minvalue, maxvalue=maxvalue)
    ax.plot(y, x, color='#880808', linestyle='-', linewidth=1.5, label=best_fit_name + ' (best fit)')

    return ax


# %% Bootstrapping
def _bootstrap(distribution, distribution_fit, X, n_boots=100, alpha=0.05, random_state=None, n_jobs=1):
    # Bootstrapping
    # the goal here is to estimate the KS statistic of the fitted distribution when the params are estimated from data.
    # 1. Resample using fitted distribution.
    # 2. Use the resampled data to fit again the distribution.
    # 3. Compare the resampled data vs. fitted PDF.

    bootstrap_score, bootstrap_pass = 0, None

    try:
        if (n_boots is not None) and (n_boots>=10):
            # Limit the number of samples to avoid memory issues.
            n = np.minimum(10000, len(X))
            # Kolmogorov-Smirnov (KS) statistic
            Dn = st.kstest(X, distribution_fit.cdf)

            def bootstrap_iteration(i):
                # Resample n times from target distribution.
                resamples = distribution_fit.rvs(n, random_state=random_state)
                # Find new target parameters after resampling
                params = distribution.fit(resamples)
                # Create new fit: k-hat
                fit = distribution(*params)
                # Score the k-hat distribution vs. for the resampled data of distribution k.
                Dn_i = st.kstest(resamples, fit.cdf)
                # Return the test statistic
                return Dn_i[0]

            Dns = Parallel(n_jobs=n_jobs)(delayed(bootstrap_iteration)(i) for i in range(n_boots))

            Dn_alpha = np.quantile(Dns, 1 - alpha)
            bootstrap_pass = False if Dn[0] > Dn_alpha else True
            # Compute ratio correct
            bootstrap_score = np.sum(Dns > Dn[0]) / n_boots
    except:
        pass
        # logger.info('[%s] > Could not estimate fit, likely due to low sample size.' %(distr))
    # Return
    return bootstrap_score, bootstrap_pass


# %% Bootstrapping
# def _bootstrap(distribution, distribution_fit, X, n_boots=100, alpha=0.05, random_state=None):
#     # Bootstrapping
#     # the goal here is to estimate the KS statistic of the fitted distribution when the params are estimated from data.
#     # 1. Resample using fitted distribution.
#     # 2. Use the resampled data to fit again the distribution.
#     # 3. Compare the resampled data vs. fitted PDF.

#     bootstrap_score, bootstrap_pass = 0, None
    
#     try:
#         if (n_boots is not None) and (n_boots>=10):
#             # Limit the number of samples to avoid memory issues.
#             n = np.minimum(10000, len(X))
#             # Kolmogorov-Smirnov (KS) statistic
#             Dn = st.kstest(X, distribution_fit.cdf)
    
#             Dns=[]
#             for i in tqdm(range(n_boots), desc="[distfit] >Bootstrapping " + distribution.name.title(), position=0, leave=False, disable=disable_tqdm()):
#                 # Resample n times from target distribution.
#                 resamples = distribution_fit.rvs(n, random_state=random_state)
#                 # Find new target parameters after resampling
#                 params = distribution.fit(resamples)
#                 # Create new fit: k-hat
#                 fit = distribution(*params)
#                 # Score the k-hat distribution vs. for the resampled data of distribution k.
#                 Dn_i = st.kstest(resamples, fit.cdf)
#                 # Store the test statistics
#                 Dns.append(Dn_i[0])
    
#             Dn_alpha = np.quantile(Dns, 1 - alpha)
#             bootstrap_pass = False if Dn[0] > Dn_alpha else True
#             # Compute ratio correct
#             bootstrap_score = np.sum(Dns > Dn[0]) / n_boots
#     except:
#         pass
#         # logger.info('[%s] > Could not estimate fit, likely due to low sample size.' %(distr))
#     # Return
#     return bootstrap_score, bootstrap_pass


# %%
def _get_properties(pdf_properties, emp_properties, bar_properties, cii_properties):
    if cii_properties is not None: cii_properties = {**{'color': '#C41E3A', 'linewidth': 3, 'linestyle': 'dashed', 'marker': 'x', 'size': 20, 'color_sign_multipletest': 'g', 'color_sign': 'g', 'color_general': 'r', 'alpha': 1}, **cii_properties}
    if emp_properties is not None: emp_properties = {**{'color': '#000000', 'linewidth': 3, 'linestyle': '-', 'label': None}, **emp_properties}
    if pdf_properties is not None: pdf_properties = {**{'color': '#880808', 'linewidth': 3, 'linestyle': '-'}, **pdf_properties}
    if bar_properties is not None: bar_properties = {**{'color': '#607B8B', 'linewidth': 1, 'edgecolor': '#5A5A5A', 'align': 'center'}, **bar_properties}
    return {'pdf': pdf_properties, 'emp': emp_properties, 'bar': bar_properties, 'cii': cii_properties}


# %%
def set_colors(df, cmap='Set1'):
    """Set colors.

    Parameters
    ----------
    df : DataFrame
        DataFrame.
    cmap : str, default: 'Set1'
        Set the colormap.

    Returns
    -------
    df : DataFrame
        DataFrame.

    """
    verbose = get_logger()
    df['color'] = colourmap.generate(df.shape[0], cmap=cmap, scheme='hex', verbose=0)
    set_logger(verbose)
    return df


# %%
def _predict(self, y):
    # Check which distribution fits best to the data
    logger.debug('Compute significance for y for the fitted theoretical distribution.')
    if not hasattr(self, 'model'): raise Exception('Error: Before making a prediction, a model must be fitted first using the function: fit_transform(X)')

    # Compute P-value for data based on null-distribution
    Pvalues = self.model['model'].cdf(y)

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
    y_proba = _do_multtest(Praw, self.multtest)
    # up/down based on threshold
    y_pred = np.repeat('none', len(y))
    if self.model['CII_max_alpha'] is not None:
        if self.bound=='up' or self.bound=='right' or self.bound=='high' or self.bound=='both':
            y_pred[y>=self.model['CII_max_alpha']]='up'
    if self.model['CII_min_alpha'] is not None:
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
def _predict_quantile(self, y):
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
def _predict_percentile(self, y):
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
        logger.debug("[%.0f] - p-value = %f" %(y[i], getstat))

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
def _plot_pdf_more(df, x, n_top, cmap, pdf_properties, ax):
    if n_top is None: n_top = 1
    if n_top>1 and (pdf_properties is not None):
        n_top = np.minimum(df.shape[0], n_top + 1)
        if cmap is not None: df['color'] = colourmap.generate(df.shape[0], cmap=cmap, scheme='hex', verbose=0)
        for i in range(1, n_top):
            # Plot pdf
            tmp_distribution = getattr(st, df['name'].iloc[i])
            tmp_y = tmp_distribution.pdf(x, loc=df['loc'].iloc[i], scale=df['scale'].iloc[i], *df['arg'].iloc[i])
            _plot_pdf(x, tmp_y, df['name'].iloc[i], {'linewidth': 2, 'linestyle': '--', 'color': df['color'].iloc[i]}, ax)


def _plot_pdf(x, y, label, pdf_properties, ax):
    if pdf_properties is not None:
        # Changing label of the pdf is not allowed.
        if pdf_properties.get('label', None) is not None: pdf_properties.pop('label')
        ax.plot(x, y, label=label, **pdf_properties)


def _plot_bar(binedges, histvals, bar_properties, ax):
    if bar_properties is not None:
        bar_properties.setdefault('width', np.diff(binedges))
        ax.bar(binedges[:-1], histvals[:-1], **bar_properties)


def _plot_emp(x, y, line_properties, ax):
    if line_properties is not None:
        line_properties.setdefault('label', 'Emperical PDF')
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
        cii_properties, cii_properties_custom = _get_cii_properties(cii_properties)
        # add CII
        ax.axvline(model['CII_min_alpha'], c=cii_properties_custom['color'], label='CII low', **cii_properties)
        ax.axvline(model['CII_max_alpha'], c=cii_properties_custom['color'], label='CII high', **cii_properties)

        # Add significant hits as line into the plot. This data is dervived from dfit.proba_parametric
        if results is not None:
            for i in range(0, len(results['y'])):
                if results['y_pred'][i] != 'none':
                    ax.axvline(results['y'][i], c=cii_properties_custom['color_sign'], **cii_properties)

            idxIN = np.logical_or(results['y_pred']=='down', results['y_pred']=='up')
            if np.any(idxIN):
                cii_properties['label']='Outside boundaries'
                ax.scatter(results['y'][idxIN], np.zeros(sum(idxIN)), color=cii_properties_custom['color_sign'], marker=cii_properties_custom['marker'], **cii_properties)
            idxOUT = results['y_pred']=='none'
            if np.any(idxOUT):
                cii_properties['label']='Inside boundaries'
                ax.scatter(results['y'][idxOUT], np.zeros(sum(idxOUT)), color=cii_properties_custom['color_general'], marker=cii_properties_custom['marker'], **cii_properties)

def _plot_cii_parametric(model, alpha, results, cii_properties, ax):
    # Collect properties
    cii_properties, cii_properties_custom = _get_cii_properties(cii_properties)

    if cii_properties is not None:
        # Plot vertical line to stress the cut-off point
        if model['CII_min_alpha'] is not None:
            cii_properties['label'] = 'CII low ' + '(' + str(alpha) + ')'
            ax.axvline(x=model['CII_min_alpha'], ymin=0, ymax=1, color=cii_properties_custom['color'], **cii_properties)
        if model['CII_max_alpha'] is not None:
            cii_properties['label'] = 'CII high ' + '(' + str(alpha) + ')'
            ax.axvline(x=model['CII_max_alpha'], ymin=0, ymax=1, color=cii_properties_custom['color'], **cii_properties)
        if cii_properties.get('label'): cii_properties.pop('label')

    # Add significant hits as line into the plot. This data is dervived from dfit.proba_parametric
    # if hasattr(self, 'results') and (cii_properties is not None):
    if (results is not None) and (cii_properties is not None):
        if alpha is None: alpha=1
        idxIN=np.where(results['y_proba']<=alpha)[0]
        logger.info("Mark %d significant regions" %(len(idxIN)))

        # Plot significant hits
        for i in idxIN:
            if cii_properties.get('label'): cii_properties.pop('label')
            ax.axvline(x=results['y'][i], ymin=0, ymax=1, markersize=cii_properties_custom['size'], marker=cii_properties_custom['marker'], color=cii_properties_custom['color_sign_multipletest'], **cii_properties)

        # Plot the samples that are not signifcant after multiple test.
        if np.any(idxIN):
            cii_properties['label'] = 'Significant'
            ax.scatter(results['y'][idxIN], np.zeros(len(idxIN)), s=50, marker=cii_properties_custom['marker'], color=cii_properties_custom['color_sign'], **cii_properties)

        # Plot the samples that are not signifcant after multiple test.
        idxOUT = np.where(results['y_proba']>alpha)[0]
        if np.any(idxOUT):
            cii_properties['label'] = 'Not significant'
            ax.scatter(results['y'][idxOUT], np.zeros(len(idxOUT)), s=50, marker=cii_properties_custom['marker'], color=cii_properties_custom['color_general'], **cii_properties)

# %% Plot
def _plot_quantile(self, title='', xlabel='Values', ylabel='Frequency', figsize=(20, 15), fontsize=16, xlim=None, ylim=None, fig=None, ax=None, grid=True, emp_properties={}, bar_properties={}, cii_properties={}):
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(grid)

    return fig, ax


# %% Plot
def _plot_parametric(self,
                     n_top=1,
                     title='',
                     figsize=(20, 15),
                     xlabel='Values',
                     ylabel='Frequency',
                     fontsize=16,
                     xlim=None,
                     ylim=None,
                     grid=True,
                     fig=None,
                     ax=None,
                     emp_properties={},
                     pdf_properties={},
                     bar_properties={},
                     cii_properties={},
                     cmap='Set1'):

    # Store output and function parameters
    model = self.model
    Param = {}
    Param['title'] = title
    Param['figsize'] = figsize
    Param['xlim'] = xlim
    Param['ylim'] = ylim
    Param['n_top'] = np.minimum(self.summary.shape[0], n_top)

    # Make figure
    best_fit_name = model['name'].title()
    arg = model['params'][:-2]
    loc = model['params'][-2]
    scale = model['params'][-1]
    distribution = getattr(st, model['name'])

    # Get pdf boundaries
    # Take maximum/minimum based on empirical data to avoid long theoretical distribution tails
    try:
        getmin = distribution.ppf(0.0000001, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.0000001, loc=loc, scale=scale)
        getmin = np.maximum(getmin, np.min(self.histdata[1]))
    except ValueError:
        logger.info('The minimum CII could not be determined. The minimum of the histogram is used instead.')
        getmin = np.min(self.histdata[1])

    try:
        getmax = distribution.ppf(0.9999999, *arg, loc=loc, scale=scale) if arg else distribution.ppf(0.9999999, loc=loc, scale=scale)
        getmax = np.minimum(getmax, np.max(self.histdata[1]))
    except ValueError:
        logger.info('The maximum CII could not be determined. The maximum of the histogram is used instead.')
        getmax = np.max(self.histdata[1])

    # Build pdf and turn into pandas Series
    x = np.linspace(getmin, getmax, self.size)
    y = distribution.pdf(x, loc=loc, scale=scale, *arg)

    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram empirical data
    _plot_bar(self.histdata[1], self.histdata[0], bar_properties, ax)
    # Plot empirical data
    _plot_emp(self.histdata[1], self.histdata[0], emp_properties, ax)
    # Plot pdf
    _plot_pdf(x, y, best_fit_name + ' (best fit)', pdf_properties, ax)
    # Plot top n pdf
    _plot_pdf_more(self.summary, x, Param['n_top'], cmap, pdf_properties, ax)
    # plot CII
    results = self.results if hasattr(self, 'results') else None
    _plot_cii_parametric(self.model, self.alpha, results, cii_properties, ax)

    # Limit axis
    if Param['xlim'] is not None:
        ax.set_xlim(Param['xlim'][0], Param['xlim'][1])
    if Param['ylim'] is not None:
        ax.set_ylim(Param['ylim'][0], Param['ylim'][1])

    ax.set_title(self._make_title(title), fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.legend(loc='upper right')
    ax.grid(grid)

    logger.info("Estimated distribution: %s(loc:%f, scale:%f)" %(model['name'].title(), model['params'][-2], model['params'][-1]))
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


def _store(alpha, stats, bins, bound, distr, histdata, method, model, multtest, n_perm, size, smooth, summary, weighted, f, n_boots, random_state):
    out = {}
    out['model'] = model
    out['summary'] = summary
    out['histdata'] = histdata
    out['size'] = size
    out['alpha'] = alpha
    out['stats'] = stats
    out['bins'] = bins
    out['bound'] = bound
    out['name'] = distr
    out['method'] = method
    out['multtest'] = multtest
    out['n_perm'] = n_perm
    out['smooth'] = smooth
    out['weighted'] = weighted
    out['f'] = f
    out['n_boots'] = n_boots
    out['random_state'] = random_state
    # Return
    return out


# %% Compute score for each distribution - in parallel when n_jobs_dist != 1
def _compute_score_distribution(data, X, y_obs, DISTRIBUTIONS, stats, cmap='Set1', n_boots=None, random_state=None, n_jobs=1, n_jobs_dist=1):
    df = pd.DataFrame(index=range(0, len(DISTRIBUTIONS)), columns=['name', 'score', 'loc', 'scale', 'arg', 'params', 'model', 'bootstrap_score', 'bootstrap_pass'])
    max_name_len = np.max(list(map(lambda x: len(x.name) if isinstance(x.name, str) else len(x.name()), DISTRIBUTIONS)))

    def fit_distribution(i, distribution):
        try:
            start_time = time.time()
            # Fit the distribution. However, this can result in an error. I need the try-except.
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                # fit dist to data
                params = distribution.fit(data)
                logger.debug(params)

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
                # Fitted model
                distribution_fit = distribution(*arg, loc, scale) if arg else distribution(loc, scale)  # Store the fitted model
                # Bootstrapping
                bootstrap_score, bootstrap_pass = _bootstrap(distribution, distribution_fit, data, n_boots=n_boots, random_state=random_state, n_jobs=n_jobs)

                # Setup for the logger - when not Parallel executed
                spaces_1 = ' ' * (max_name_len - len(distr_name))
                scores = ('[%s: %g] [loc=%.3f scale=%.3f]' % (stats, score, loc, scale))
                time_spent = time.time() - start_time
                logger.info("[%s%s] [%.4s sec] %s" % (distr_name, spaces_1, time_spent, scores))

                return (i, distr_name, score, loc, scale, arg, params, distribution_fit, bootstrap_score, bootstrap_pass, start_time)

        except Exception as e:
            return None

    # Parallelize the loop over distributions
    results = Parallel(n_jobs=n_jobs_dist)(
        delayed(fit_distribution)(i, distribution) for i, distribution in enumerate(DISTRIBUTIONS)
    )

    # Fill the DataFrame with the results
    for result in results:
        if result is not None:
            i, distr_name, score, loc, scale, arg, params, distribution_fit, bootstrap_score, bootstrap_pass, start_time = result
            df.values[i] = [distr_name, score, loc, scale, arg, params, distribution_fit, bootstrap_score, bootstrap_pass]

            # Setup for the logger
            spaces_1 = ' ' * (max_name_len - len(distr_name))
            scores = ('[%s: %g] [loc=%.3f scale=%.3f]' % (stats, score, loc, scale))
            time_spent = time.time() - start_time
            logger.info("[%s%s] [%.4s sec] %s" % (distr_name, spaces_1, time_spent, scores))

    # Sort the output
    df, model = _sort_dataframe(df, cmap=cmap)
    # Return
    return df, model


# %% Sort dataframe
def _sort_dataframe(df, cmap='Set1'):
    # df = df.sort_values('score')
    df.sort_values(by=['bootstrap_score', 'score', 'bootstrap_pass'], ascending=[False, True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = set_colors(df, cmap=cmap)
    # Get best model
    model = df.iloc[0,:].to_dict()

    return df, model


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
        raise Exception('[distfit] [%] statistic not implemented.', stats)
    return score


# %% Determine confidence intervals on the best fitting distribution
def compute_cii(self, model, alpha=None, logger=None):
    CIIup, CIIdown = None, None
    if alpha is None: alpha = self.alpha

    if logger is not None: logger.info("Compute confidence intervals [%s]" %(self.method))
    if self.method not in ['parametric', 'quantile', 'percentile', 'discrete']:
        raise Exception('[distfit] >Error: method parameter can only be of type: "parametric", "quantile", "percentile" or "discrete".')

    if (self.method=='parametric') or (self.method=='discrete'):
        # Determine %CII
        if alpha is not None:
            _, CIIdown, flag1 = get_ppf(self, model, bound=self.bound, alpha=alpha, logger=logger)
            CIIup, _, flag2 = get_ppf(self, model, bound=self.bound, alpha=alpha, logger=logger)
            if (flag1 or flag2): logger.info('The CII with alpha=%g could not be determined for [%s]. Try different values for alpha. The minimum of the histogram is used instead.' %(alpha, model['name']))

            # if self.bound=='up' or self.bound=='both' or self.bound=='right' or self.bound=='high':
            #     CIIdown = model['model'].ppf(1 - alpha)
            # if self.bound=='down' or self.bound=='both' or self.bound=='left' or self.bound=='low':
            #     CIIup = model['model'].ppf(alpha)
    elif self.method=='quantile':
        X = model
        model = {}
        CIIdown = np.quantile(X, 1 - alpha)
        CIIup = np.quantile(X, alpha)
    elif self.method=='percentile':
        X = model
        model = {}
        # Set Confidence intervals
        cii_high = (0 + (alpha / 2)) * 100
        cii_low = (1 - (alpha / 2)) * 100
        CIIup = np.percentile(X, cii_high)
        CIIdown = np.percentile(X, cii_low)

    # Store
    model['CII_min_alpha'] = CIIup
    model['CII_max_alpha'] = CIIdown
    return model


# Get the CII boundaries
def get_ppf(self, model, bound, alpha, logger=None):
    CIIup, CIIdown = None, None
    flag=False
    if self.bound=='up' or self.bound=='both' or self.bound=='right' or self.bound=='high':
        try:
            CIIdown = model['model'].ppf(1 - alpha)
        except ValueError as e:
            flag=True
            logger.debug(e)
            CIIdown = np.max(self.histdata[1])

    if self.bound=='down' or self.bound=='both' or self.bound=='left' or self.bound=='low':
        try:
            CIIup = model['model'].ppf(alpha)
        except ValueError as e:
            flag=True
            logger.debug(e)
            CIIup = np.min(self.histdata[1])

    return CIIup, CIIdown, flag


# Multiple test correction
def _do_multtest(Praw, multtest='fdr_bh'):
    """Multiple test correction for input pvalues.

    Parameters
    ----------
    Praw : list of float
        Pvalues.
    multtest : str, default: 'fdr_bh'
        Multiple testing method. Options are:
            * None : No multiple testing
            * 'bonferroni' : one-step correction
            * 'sidak' : one-step correction
            * 'holm-sidak' : step down method using Sidak adjustments
            * 'holm' : step-down method using Bonferroni adjustments
            * 'simes-hochberg' : step-up method  (independent)
            * 'hommel' : closed method based on Simes tests (non-negative)
            * 'fdr_bh' : Benjamini/Hochberg  (non-negative)
            * 'fdr_by' : Benjamini/Yekutieli (negative)
            * 'fdr_tsbh' : two stage fdr correction (non-negative)
            * 'fdr_tsbky' : two stage fdr correction (non-negative)

    Returns
    -------
    list of float.
        Corrected pvalues.

    """
    if not isinstance(multtest, type(None)):
        logger.info("Multiple test correction method applied: [%s]." %multtest)
        logger.debug(Praw)
        Padj = multitest.multipletests(Praw, method=multtest)[1]
    else:
        Padj=Praw

    Padj = np.clip(Padj, 0, 1)
    return Padj


def smoothline(xs, ys=None, interpol=3, window=1, verbose=None):
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
    if verbose is not None: set_logger(verbose)
    if window is not None:
        logger.info('[smoothline] >Smoothing by interpolation..')
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


def transform_binom(hist, plot=True, weighted=True, f=1.5, stats='RSS'):
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
    logger.debug(f'[distfit] >Initial estimate: n={nest}, p={mean/nest:.3g}')

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
        logger.debug('[binomial] [%s=%.3g] Trying n=%s -> p=%.3g, (initial=%.3g)' %(stats, score, nval, p, p_guess))

    n_fit = np.argmin(scores) + nmin
    p_fit = pvals[n_fit - nmin]
    score = scores[n_fit - nmin]
    chi2r = score / (nk - 2) if nk > 2 else np.nan
    logger.info('[distfit] >[binomial] [%s=%.3g] [n=%.2g] [p=%.6g] [chi^2=%.3g]' %(stats, score, n_fit, p_fit, chi2r))

    # Store
    model = {}
    model['name'] = 'binom'
    # model['name'] = st.binom
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


def fit_transform_binom(X, f=1.5, weighted=True, stats='RSS'):
    """Convert array of samples (nonnegative ints) to histogram and fit."""
    logger.info('Fit using binomial distribution.')
    hist = fit_binom(X)
    model, figdata = transform_binom(hist, f=f, weighted=weighted, stats=stats)

    # Create dataframe (this is required to be consistent with other parts)
    df = pd.DataFrame(index=range(0, 1), columns=['name', 'score', 'loc', 'scale', 'arg', 'params', 'model'])
    df['name'] = model['name']
    # df['name'] = model['name']
    df['model'] = model['model']  # Store the fitted model
    df['params'] = [model['params']]
    df['score'] = model['score']
    df['loc'] = model['n']
    df['scale'] = model['p']
    df['arg'] = None
    return df, model, figdata


def plot_binom(self,
               emp_properties={},
               pdf_properties={},
               bar_properties={},
               cii_properties={},
               fontsize=16,
               xlabel='Values',
               ylabel='Frequency',
               title='',
               figsize=(20, 15),
               xlim=None,
               ylim=None,
               grid=True,
               ):
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
    # dfit = self.model['name']
    best_fit_name = self.model['name'].title()
    best_fit_param = self.model['params']
    cii_properties, cii_properties_custom = _get_cii_properties(cii_properties)

    model = self.model
    # figdata = self.figdata
    n_fit = model['n']
    p_fit = model['p']
    histf = BinomPMF(n_fit)(self.figdata['Xdata'], p_fit) * self.figdata['hist'].sum()

    # Init figure
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    # plot bar
    if bar_properties is not None:
        bar_properties['align']='center'
        bar_properties['label']='Histogram'
        ax[0].bar(self.figdata['Xdata'], self.figdata['hist'], **bar_properties)
    # plot Emperical data
    if emp_properties is not None:
        ax[0].plot(self.figdata['Xdata'], self.figdata['hist'], 'o', color=emp_properties['color'], label=emp_properties['label'])

    # plot PDF
    if pdf_properties is not None:
        pdf_properties['label'] = 'PMF (binomial)'
        ax[0].step(self.figdata['Xdata'], histf, where='mid', **pdf_properties)
        ax[0].axhline(0, color=pdf_properties['color'])

    # Plot CII
    if cii_properties is not None:
        # Plot vertical line to stress the cut-off point
        if self.model['CII_min_alpha'] is not None:
            cii_properties['label'] = 'CII low ' + '(' + str(self.alpha) + ')'
            ax[0].axvline(x=model['CII_min_alpha'], ymin=0, ymax=1, color=cii_properties_custom['color'], **cii_properties)
        if self.model['CII_max_alpha'] is not None:
            cii_properties['label'] = 'CII high ' + '(' + str(self.alpha) + ')'
            ax[0].axvline(x=model['CII_max_alpha'], ymin=0, ymax=1, color=cii_properties_custom['color'], **cii_properties)

        # Add significant hits as line into the plot. This data is dervived from dfit.proba_parametric
        if hasattr(self, 'results'):
            # Plot significant hits with multiple test
            if self.alpha is None: self.alpha=1
            idxIN=np.where(self.results['y_proba']<=self.alpha)[0]
            logger.debug("[distfit] >Plot Number of significant regions detected: %d" %(len(idxIN)))
            if cii_properties.get('label'): cii_properties.pop('label')
            for i in idxIN:
                ax[0].axvline(x=self.results['y'][i], ymin=0, ymax=1, markersize=cii_properties_custom['size'], marker=cii_properties_custom['marker'], color=cii_properties_custom['color_sign_multipletest'], **cii_properties)

            # Plot the samples that signifcant without multiple test.
            if np.any(idxIN):
                cii_properties['label']='Significant'
                ax[0].scatter(self.results['y'][idxIN], np.zeros(len(idxIN)), s=cii_properties_custom['size'], marker=cii_properties_custom['marker'], color=cii_properties_custom['color_sign'], **cii_properties)

            # Plot the samples that are not signifcant.
            idxOUT = np.where(self.results['y_proba']>self.alpha)[0]
            if np.any(idxOUT):
                cii_properties['label']='Not significant'
                ax[0].scatter(self.results['y'][idxOUT], np.zeros(len(idxOUT)), s=cii_properties_custom['size'], marker=cii_properties_custom['marker'], color=cii_properties_custom['color_general'], **cii_properties)

    # Limit axis
    if Param['xlim'] is not None:
        ax[0].set_xlim(xmin=Param['xlim'][0], xmax=Param['xlim'][1])
    if Param['ylim'] is not None:
        ax[0].set_ylim(ymin=Param['ylim'][0], ymax=Param['ylim'][1])
    ax[0].set_xlabel(xlabel, fontsize=fontsize)
    ax[0].set_ylabel(ylabel, fontsize=fontsize)
    ax[0].legend(loc='upper right')
    ax[0].grid(grid)
    param_str = ', '.join(['{}={:g}'.format(k, v) for k, v in zip(['n', 'p'], best_fit_param)])
    ax[0].set_title('%s\n%s\n%s' %(Param['title'], best_fit_name, param_str), fontsize=fontsize)
    ax[0].legend(loc='upper right')
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0].grid(grid)

    # Second image
    ax[1].set_xlabel('n', fontsize=fontsize)
    ax[1].set_ylabel(self.stats, fontsize=fontsize)
    plotfunc = ax[1].semilogy if self.figdata['scores'].max()>20 * self.figdata['scores'].min()>0 else ax[1].plot
    plotfunc(self.figdata['nvals'], self.figdata['scores'], 'k-', label=('%s over n scan' %self.stats))
    ax[1].vlines(n_fit, 0, self.figdata['scores'].max(), color=cii_properties_custom['color'], linestyles='dashed')
    ax[1].hlines(model['score'], self.figdata['nvals'].min(), self.figdata['nvals'].max(), color=cii_properties_custom['color'], linestyles='dashed', label="Best %s: %.3g" %(self.stats, model['score']))
    ax[1].legend(loc='upper right')
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1].grid(grid)
    fig.show()

    logger.debug("[distfit] Estimated distribution: %s [loc:%f, scale:%f]" %(model['name'], model['params'][-2], model['params'][-1]))
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

def scale_data_minmax(X, minvalue, maxvalue):
    return (((X - min(X)) / (max(X) - min(X))) * (maxvalue - minvalue)) + minvalue

def scale_data(y):
    return [(x - min(y)) / (max(y) - min(y)) for x in y]
    # ynorm = y
    # for i, value in enumerate(y):
    #     ynorm[i] = (value - min(y)) / (max(y) - min(y))
    # return ynorm
    
# %%
def set_logger(verbose: [str, int] = 'info'):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
            * 0, 60, None, 'silent', 'off', 'no']: No message.
            * 10, 'debug': Messages from debug level and higher.
            * 20, 'info': Messages from info level and higher.
            * 30, 'warning': Messages from warning level and higher.
            * 50, 'critical': Messages from critical level and higher.

    Returns
    -------
    None.

    Examples
    --------
    >>> # Set the logger to warning
    >>> set_logger(verbose='warning')
    >>>
    >>> # Test with different messages
    >>> logger.debug("Hello debug")
    >>> logger.info("Hello info")
    >>> logger.warning("Hello warning")
    >>> logger.critical("Hello critical")
    >>>
    """
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {'silent': 60,
                  'off': 60,
                  'no': 60,
                  'debug': 10,
                  'info': 20,
                  'warning': 30,
                  'critical': 50}
        verbose = levels[verbose]

    # Show examples
    logger.setLevel(verbose)


# %%
def get_logger():
    return logger.getEffectiveLevel()


# %%
# def disable_tqdm():
#     """Set the logger for verbosity messages."""
#     return (True if (logger.getEffectiveLevel()>=30) else False)

def check_version():
    import matplotlib
    if version.parse(matplotlib.__version__) < version.parse('3.5.2'):
        logger.error('This release requires matplotlib version >= 3.5.2. Try: pip install -U matplotlib')

