from distfit.distfit import distfit
from scatterd import scatterd

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, multivariate_normal
import scipy.stats as st
from scipy.interpolate import griddata

import logging
logger = logging.getLogger(__name__)

class Multidistfit:
    def __init__(self, model):
        """Multivariate distribution fitting.

        X (n × d)
         ↓
        fit_marginals()
         ↓
        transform_to_uniform()
         ↓
        fit_dependence()
         ↓
        joint_pdf()

        """
        self.model = model
        
    def fit_transform(self, X):
        logger.info('Multivariate probability density fitting.')
        # Jitter regularization to prevent small number issues
        X = regularize_covariance(X)
        # Fit independent univariate distributions per feature.
        self.marginals = self.fit_marginals(X)
        # Transform data into uniform space using fitted marginals.
        self.U = transform_to_uniform(X, self.marginals)
        # Store empirical correlation
        self.corr = fit_dependence(self.U)
        # Compute joint pdf vector
        self.joint_pdf = compute_joint_pdf(X, self.marginals, self.U, self.corr)

    def fit_marginals(self, X):
        # Fit independent univariate distributions per feature.
        logger.info('Fitting marginals')
        marginals = {}
    
        for i in range(X.shape[1]):
            model = distfit(method=self.model.method, 
                            alpha = self.model.alpha,
                            bins = self.model.bins,
                            bound = self.model.bound,
                            distr = self.model.distr,
                            n_boots = self.model.n_boots,
                            multtest = self.model.multtest,
                            smooth = self.model.smooth,
                            n_perm = self.model.n_perm,
                            stats = self.model.stats,
                            todf = self.model.todf,
                            f = self.model.f,
                            weighted = self.model.weighted,
                            mhist = self.model.mhist,
                            random_state = self.model.random_state,
                            verbose = self.model.verbose,
                            )
            
            model.fit_transform(X[:, i])
            marginals[i] = model
    
        return marginals
    

    def generate(self, n=1000, random_state=None):
        """Generate multivariate synthetic data based on the multidistribution fit.

        Parameters
        ----------
        n : int (default: 1000)
            Number of samples to generate.
        random_state : [int, None] (default: None)
            Random state.

        Returns
        -------
        X : np.array
           Numpy array with generated data.

        Examples
        --------
        >>> # Import library
        >>> from distfit import distfit
        >>>
        >>> # Initialize with multivariate=True and other custom parameters.
        >>> dfit = distfit(multivariate=True)
        >>>
        >>> Get example data
        >>> X = dfit.import_example(data='multi_t')
        >>>
        >>> # Fit model
        >>> dfit.fit_transform(X)
        >>>
        >>> # Create syntethic data using fitted distribution.
        >>> Xnew = dfit.generate(10)
        >>>

        """
        rng = np.random.default_rng(random_state)
        d = len(self.marginals)
        Z = rng.multivariate_normal(mean=np.zeros(d), cov=self.corr, size=n)
        U = norm.cdf(Z)
        X = np.zeros_like(U)

        for i, model in self.marginals.items():
            X[:, i] = model.model['model'].ppf(U[:, i])

        return X

    def predict_outliers(self, X):
        """
        Identify outliers based on low joint likelihood under the fitted copula model.

        Samples are flagged as outliers if their joint log-density falls below
        a specified percentile threshold of the empirical log-density distribution.
        This method detects multivariate outliers, including dependency violations
        that are not visible in marginal distributions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for which outliers should be detected. Each row represents one multivariate observation.

        Returns
        -------
        outliers : ndarray of shape (n_samples,)
            Boolean array where ``True`` indicates that the corresponding sample is classified as an outlier based on low joint likelihood.

        Notes
        -----
        - The joint density is evaluated using the fitted marginal distributions and the copula-based dependence structure.
        - The returned values are based on *relative* likelihoods; the density values themselves are not probabilities.
        - Percentile-based thresholding is scale-free and robust to dimensionality.

        See Also
        --------
        evaluate_pdf : Evaluate the joint probability density of samples.
        score        : Compute average log-likelihood of samples.

        """
        p = self.evaluate_pdf(X)
        logp = np.log(p['copula_density'])
        outliers = logp < np.percentile(logp, 1)
        return outliers

    def evaluate_pdf(self, X):
        """
        Evaluate the joint density of samples using a Gaussian copula model.

        This method computes the joint probability density by combining marginal probability density functions with a Gaussian copula
        that captures the dependence structure between variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for which the joint density is evaluated. Each row represents one multivariate observation.

        Returns
        -------
        out : dict
            Dictionary containing:

            - ``'copula_density'`` : ndarray of shape (n_samples,)
                Joint density values evaluated at each sample. These values
                represent relative likelihoods and are not probabilities.

            - ``'score'`` : float
                Mean log joint density across all samples. This can be used
                for model comparison or goodness-of-fit assessment.

        Notes
        -----
        - Marginal distributions are evaluated independently and transformed
          to uniform variables via their cumulative distribution functions (CDFs).
        - The uniform variables are mapped to standard normal space using the
          inverse Gaussian CDF (probability integral transform).
        - The Gaussian copula density is computed as the ratio between the
          multivariate normal density and the product of univariate normal densities.
        - The resulting joint density is invariant to monotonic transformations
          of the marginals.

        See Also
        --------
        predict_outliers : Detect multivariate outliers using joint log-density.
        """
        X = np.atleast_2d(X)
        d = X.shape[1]

        # Marginal CDFs and PDFs
        U = np.zeros_like(X)
        log_pdf = np.zeros(len(X))

        for i, m in enumerate(self.marginals):
            dist = self.marginals[m].model['model']
            U[:, i] = dist.cdf(X[:, i])
            log_pdf += np.log(dist.pdf(X[:, i]))

        # Convert to Gaussian space
        Z = norm.ppf(U)

        # Copula density
        mvn = multivariate_normal(mean=np.zeros(d), cov=self.corr)
        log_copula = (mvn.logpdf(Z) - np.sum(norm.logpdf(Z), axis=1))
        p = np.exp(log_copula + log_pdf)

        # Score for model comparison
        score = np.mean(np.log(p))
        out = {'score': score, 'copula_density': p}
        return out

# %%
# =============================================================================
# HELPERS
# =============================================================================
def transform_to_uniform(X, marginals):
    # Probability Integral Transform (PIT) to transform data into uniform space using fitted marginals.
    logger.info('Probability Integral Transform (PIT)')
    U = np.zeros_like(X, dtype=float)

    for i, model in marginals.items():
        U[:, i] = model.model['model'].cdf(X[:, i])

    return U

def fit_dependence(U):
    logger.info('Dependence Modeling: storing empirical correlations')
    return np.corrcoef(U.T)

def regularize_covariance(X, eps=1e-6):
    # jitter regularization
    logger.info('Jitter regularization')
    return X + eps * np.random.randn(*X.shape)

def compute_joint_pdf(X, marginals, U, corr):
    # Probability Integral Transform (PIT)
    # p(x)=c(u)i∏​fi​(xi​)
    logger.info('Computing the joint PDF vector.')

    Z = np.quantile(np.random.normal(size=10_000), U)
    copula = multivariate_normal(cov=corr)
    copula_density = copula.pdf(Z)

    marginal_density = np.ones(X.shape[0])
    for i, model in marginals.items():
        marginal_density *= model.model['model'].pdf(X[:, i])

    return copula_density * marginal_density

# %%
# =============================================================================
# Create example datasets
# =============================================================================
def make_example_dataset(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    corr = np.array([
        [1.0, 0.7, 0.4, 0.2],
        [0.7, 1.0, 0.5, 0.3],
        [0.4, 0.5, 1.0, 0.6],
        [0.2, 0.3, 0.6, 1.0],
    ])

    Z = rng.multivariate_normal(mean=np.zeros(4), cov=corr, size=n)
    U = st.norm.cdf(Z)

    X0 = st.gamma(a=2.0).ppf(U[:, 0])
    X1 = st.lognorm(s=0.6).ppf(U[:, 1])
    X2 = st.beta(a=2.0, b=5.0).ppf(U[:, 2])
    X3 = st.norm(loc=0, scale=1).ppf(U[:, 3])

    X = np.column_stack([X0, X1, X2, X3])
    return X

def make_heavy_tail_dataset(n=2000, seed=42, df=3):
    rng = np.random.default_rng(seed)

    corr = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0],
    ])

    Z = rng.multivariate_normal(np.zeros(3), corr, size=n)
    T = st.t(df=df).cdf(Z)

    X0 = st.gamma(a=2).ppf(T[:, 0])
    X1 = st.lognorm(s=0.7).ppf(T[:, 1])
    X2 = st.norm().ppf(T[:, 2])

    return np.column_stack([X0, X1, X2])

# %%
# =============================================================================
# PLOTS
# =============================================================================
def _plot_copula(
    U,
    bins=30,
    figsize=(10, 6),
    title="",
    properties=None,
    legend=True,
    fig=None,
    ax=None,
    ):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    # Default bar styling
    default = {"color": "#607B8B", "edgecolor": "#5A5A5A", "linewidth": 1, "alpha": 0.85, "align": "center"}
    properties = {**default, **properties}
    if properties is not None and properties.get('legend', None) is not None:
        properties.pop('legend')

    # Histogram values
    histvals, binedges = np.histogram(U, bins=bins, range=(0, 1), density=True)
    bin_centers = 0.5 * (binedges[:-1] + binedges[1:])
    bin_width = binedges[1] - binedges[0]  # Calculate the width of each bin

    # Plot histogram using bars
    ax.bar(bin_centers, histvals, width=bin_width, **properties)

    # Uniform reference line
    ax.hlines(
        1,
        xmin=0,
        xmax=1,
        colors="black",
        linestyles="--",
        linewidth=2,
        label="Uniform(0,1)",
    )

    # Styling
    ax.set_xlim(0, 1)
    ax.set_title(f"PIT Uniformity for {title}", fontsize=18)
    ax.set_xlabel("u", fontsize=26)
    ax.set_ylabel("Density", fontsize=26)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if legend: ax.legend(frameon=False, fontsize=14)
    # Return
    return fig, ax


def _plot_dependence_copula(U, plot_type='uniform', figsize=None, properties={}, title='', verbose='info'):
    if plot_type=='uniform':
        if title is None: title="Dependence (uniform) Copula Space"
        label = 'U'
        X = U
    elif plot_type=='gaussian':
        if title is None: title="Gaussian Copula Space"
        label = 'Z'
        X = norm.ppf(U)
    else:
        logger.warning(f'No valid plot_type: {plot_type}')
        return None, None

    # Default scatter styling
    default = {"s": 50, "alpha": 0.8, "c": [0.290, 0.486, 0.619], "edgecolor": 'white'}
    properties = {**default, **properties}
    if properties is not None and properties.get('legend', None) is not None:
        properties.pop('legend')


    d = X.shape[1]
    n_pairs = d * (d - 1) // 2

    # Calculate grid dimensions
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    # Auto-scale figure size
    if figsize is None: figsize = (5 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each pair
    pair_idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            ax = axes[pair_idx]

            # Scatter plot
            scatterd(X[:, i], X[:, j], ax=ax, **properties, verbose=verbose)

            # Reference box for uniform copula
            if plot_type=='uniform_copula':
                ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', linewidth=1, alpha=0.3)

            # Styling
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(f"${label}_{{{i+1}}}$", fontsize=26)
            ax.set_ylabel(f"${label}_{{{j+1}}}$", fontsize=26)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.set_title(f"{title}", fontsize=18)
            pair_idx += 1

    # Hide unused subplots
    for idx in range(pair_idx, len(axes)): axes[idx].axis('off')
    fig.tight_layout()

    # Return
    return fig, axes

def _plot_joint_density(
    X,
    pdf_vals,
    i,
    j,
    ax,
    gridsize=50,
    method="linear"
):
    xi = np.linspace(X[:, i].min(), X[:, i].max(), gridsize)
    xj = np.linspace(X[:, j].min(), X[:, j].max(), gridsize)
    xx, yy = np.meshgrid(xi, xj)

    zz = griddata(
        points=X[:, [i, j]],
        values=pdf_vals,
        xi=(xx, yy),
        method=method
    )

    cs = ax.contourf(xx, yy, zz, levels=20, cmap="viridis")
    ax.set_xlabel(f"X{i}")
    ax.set_ylabel(f"X{j}")

    return cs

def _plot_joint_pairplot(
    X,
    pdf_vals,
    marginals,
    gridsize=40,
    figsize=None
):
    """
    Pairplot-style grid of joint density slices.
    """
    d = X.shape[1]
    if figsize is None:
        figsize = (7 * d, 5 * d)

    fig, axes = plt.subplots(d, d, figsize=figsize)
    contour_ref = None

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if i == j:
                marginals[i].plot(ax=ax, legend=False, title=None)
            else:
                contour_ref = _plot_joint_density(
                    X,
                    pdf_vals,
                    i=j,
                    j=i,
                    ax=ax,
                    gridsize=gridsize
                )

            if i < d - 1:
                ax.set_xlabel("")
            if j > 0:
                ax.set_ylabel("")

    # Single shared colorbar
    if contour_ref is not None:
        fig.colorbar(
            contour_ref,
            ax=axes,
            shrink=0.6,
            label="Joint density"
        )

    fig.suptitle("Joint density pairplot (slice-based)", fontsize=16)
    # fig.tight_layout()
    return fig, axes


def pairplot_copula_uniform(
        U,
        bins=30,
        s=10,
        color=[0.4, 0.4, 0.5],
        figsize=None,
        properties=None,
        legend=True,
        fig=None,
        ax=None,
        ):
    """
    Pairplot in copula (U) space.

    Diagonal: PIT uniformity histograms
    Off-diagonal: pairwise copula scatter plots
    """
    d = U.shape[1]
    if figsize is None:
        figsize = (7 * d, 5 * d)
    if properties is not None and properties.get('legend', None) is not None:
        properties.pop('legend')

    fig, axes = plt.subplots(d, d, figsize=figsize)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if i == j:
                # Use existing uniformity function
                _plot_copula(
                    U[:, i],
                    bins=bins,
                    title=f"U{i}",
                    legend=False,
                    properties=properties,
                    fig=fig,
                    ax=ax,
                )
            else:
                scatterd(U[:, j], U[:, i], s=s, c=color, alpha=0.4, ax=ax, legend=False)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            # Clean pairplot look
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Copula Space Pairplot (U-space)", fontsize=16)
    fig.tight_layout()

    return fig, axes

# =============================================================================
# todo
# =============================================================================

def pairplot_copula_gaussian(
        U,
        bins=30,
        s=10,
        color=[0.4, 0.4, 0.5],
        figsize=None,
        properties=None,
        legend=True,
        fig=None,
        ax=None,
    ):
    """
    Pairplot in Gaussian copula (Z) space.

    Diagonal: marginal normality checks
    Off-diagonal: Gaussian copula dependence
    """
    Z = norm.ppf(U)
    d = Z.shape[1]

    if figsize is None:
        figsize = (7 * d, 5 * d)

    if properties is not None and properties.get('legend', None) is not None:
        properties.pop('legend')

    fig, axes = plt.subplots(d, d, figsize=figsize)

    # Global symmetric limits for Z-space
    zlim = np.nanmax(np.abs(Z)) * 1.05

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if i == j:
                # Diagonal: normality check
                ax.hist(
                    Z[:, i],
                    bins=bins,
                    density=True,
                    color=color,
                    alpha=0.85,
                    edgecolor="white"
                )

                x = np.linspace(-zlim, zlim, 300)
                ax.plot(x, norm.pdf(x), "k--", linewidth=2, label="N(0,1)")

                if legend:
                    ax.legend(frameon=False)

            else:
                scatterd(
                    Z[:, j],
                    Z[:, i],
                    s=s,
                    c=color,
                    alpha=0.4,
                    ax=ax,
                    legend=False
                )

                ax.set_xlim(-zlim, zlim)
                ax.set_ylim(-zlim, zlim)

            # Clean pairplot look
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Gaussian Copula Pairplot (Z-space)", fontsize=16)
    fig.tight_layout()

    return fig, axes
