# %%
# --------------------------------------------------
# Name        : examples.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/distfit
# --------------------------------------------------

import numpy as np
import distfit
print(distfit.__version__)
print(dir(distfit))

# %% Import class
from distfit import distfit

# %%
# from sklearn.datasets.samples_generator import make_blobs
# [data, labels_true] = make_blobs(n_samples=10000, centers=3, n_features=1, cluster_std=0.3, random_state=0)


# %%
X = np.random.normal(0, 2, 5000)
y = [-8,-6,0,1,2,3,4,5,6]

dist = distfit(distr='full')
model = dist.fit_transform(X)
dist.plot()

# Make prediction
results = dist.predict(y)

# plot
dist.plot()
dist.plot_summary()

# Save
dist.save(filepath='c:\\temp\\model.pkl')
# Load
dist.load(filepath='c:\\temp\\model.pkl')

# Store entire object


# %%
X = np.random.normal(0, 2, 100)
model = distfit(smooth=10)
model.fit_transform(X)
model.plot()

# %%
# Create random data with varying number of samples
import pandas as pd
samples = np.arange(250, 20000, 250)

#%%
# Initialize model
import matplotlib.pyplot as plt
from tqdm import tqdm
smooth_window=[None,1,3,5,7,9,11]
plt.figure(figsize=(15,10))

for smooth in tqdm(smooth_window):
    dist = distfit(distr='norm', smooth=smooth)
    # Estimate paramters for the number of samples
    out = []
    for s in samples:
        X = np.random.normal(0, 2, s)
        dist.fit_transform(X, verbose=0)
        out.append([dist.model['loc'], dist.model['scale'], dist.model['name'], np.where(dist.summary['distr']=='norm')[0][0], s])

    df=pd.DataFrame(out, columns=['std','mu','name','norm_place','samples'])
    ax=df['mu'].plot(grid=True, label='smooth: '+str(smooth) + ' - ' + str(df['mu'].mean()))

ax.set_xlabel('Nr.Samples')
ax.set_ylabel('mu')
ax.set_xticks(np.arange(0,len(samples)))
ax.set_xticklabels(samples.astype(str))
ax.legend()

# ax=df['std'].plot(grid=True)
# ax.set_xlabel('Nr.Samples')
# ax.set_ylabel('std')
# ax.set_xticks(np.arange(0,len(samples)))
# ax.set_xticklabels(samples.astype(str))

# %% Fit and transform
X = np.random.beta(5, 8, [100,100])
y = [-1,-0.8,-0.6,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.5]

dist = distfit()
dist.fit()
dist.transform(X)
dist.plot()
dist.predict(y)
dist.plot()

dist.plot_summary()

# %%  for Fit and transform in one go
X = np.random.beta(5, 8, [100,100])
y = [-1,-0.8,-0.6,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.5]

model = distfit()
model.fit_transform(X)
model.plot()
model.predict(y)
model.plot()

model.plot_summary()

# %% Show some results
print(model.y_proba)
print(model.y_pred)
print(model.df)
print(model.summary)


# %%
X = np.random.normal(0, 2, 1000)
y = [-8,-6,0,1,2,3,4,5,6]

model = distfit()
model.fit_transform(X)
model.plot()

# Make prediction
model.predict(y)
model.plot()

# %%
X = np.random.normal(5, 8, [100,100])
y = [-35, -10, 0, 10, 15, 35]

model = distfit()
model.fit_transform(X)
model.predict(y)
model.plot()

model = distfit()
model.fit_transform(X)
model.predict(y)
model.plot()

model.y_proba
model.y_pred
model.df

# %%
X = np.random.beta(5, 8, 1000)

model = distfit()
model.fit_transform(X)
model.plot()

# %% Find distribution parameters
X = np.random.normal(0, 2, 5000)
model = distfit()
model.fit_transform(X)
model.plot()

X = np.random.normal(10, 1, 5000)
model = distfit()
model.fit_transform(X)
model.plot()

X = np.random.normal(10, 5, 5000)
model = distfit()
model.fit_transform(X)
model.plot()

# %%
X = np.random.normal(0, 2, 1000)
y = [-8,-6,0,1,2,3,4,5,6]

model = distfit()
model.fit_transform(X)
model.predict(y)
model.plot()
model.df

model = distfit(bound='up')
model.fit_transform(X)
model.predict(y)
model.plot()
model.df

model = distfit(bound='down')
model.fit_transform(X)
model.predict(y)
model.plot()
model.df

# %% Find best fit distribution
X = np.random.normal(0, 2, 1000)
y = [-8,-6,0,1,2,3,4,5,6]

model = distfit()
model.fit_transform(X)
model.plot()

model = distfit(distr='popular')
model.fit_transform(X)
model.plot()

model = distfit(distr='full')
model.fit_transform(X)
model.plot()


# %% Quantile approach
from distfit import distfit
import numpy as np

X = np.random.normal(10, 3, 2000)
y = [3,4,5,6,10,11,12,18,20]

# Initialize
dist = distfit(method='percentile', alpha=0.05)
# dist = distfit(method='quantile', alpha=0.05)
dist.fit_transform(X)
dist.plot()

# Make prediction
dist.predict(y)
dist.plot()
dist.plot_summary()

from tabulate import tabulate
print(tabulate(dist.df, tablefmt="grid", headers="keys"))

# %% TODO

# I summarize the question as: given a list of nonnegative integers, can we fit a probability distribution, in particular a Gaussian, multinomial, and Bernoulli, and compare the quality of the fit?

# For discrete quantities, the correct term is probability mass function: P(k) is the probability that a number picked is exactly equal to the integer value k. A Bernoulli distribution can be parametrized by a p parameter: Be(k, p) where 0 <= p <= 1 and k can only take the values 0 or 1. It is a special case of the binomial distribution B(k, p, n) that has parameters 0 <= p <= 1 and integer n >= 1. (See the linked Wikipedia article for an explanation of the meaning of p and n) It is related to the Bernoulli distribution as Be(k, p) = B(k, p, n=1). The trinomial distribution T(k1, k2, p1, p2, n) is parametrized by p1, p2, n and describes the probability of pairs (k1, k2). For example, the set {(0,0), (0,1), (1,0), (0,1), (0,0)} could be pulled from a trinomial distribution. Binomial and trinomial distributions are special cases of multinomial distributions; if you have data occuring as quintuples such as (1, 5, 5, 2, 7), they could be pulled from a multinomial (hexanomial?) distribution M6(k1, ..., k5, p1, ..., p5, n). The question specifically asks for the probability distribution of the numbers of a single column, so the only multinomial distribution that fits here is the binomial one, unless you specify that the sequence [0, 1, 5, 2, 3, 1] should be interpreted as [(0, 1), (5, 2), (3, 1)] or as [(0, 1, 5), (2, 3, 1)]. But the question does not specify that numbers can be accumulated in pairs or triplets.

# Therefore, as far as discrete distributions go, the PMF for one list of integers is of the form P(k) and can only be fitted to the binomial distribution, with suitable n and p values. If the best fit is obtained for n=1, then it is a Bernoulli distribution.

# The Gaussian distribution is a continuous distribution G(x, mu, sigma), where mu (mean) and sigma (standard deviation) are parameters. It tells you that the probability of finding x0-a/2 < x < x0+a/2 is equal to G(x0, mu, sigma)*a, for a << sigma. Strictly speaking, the Gaussian distribution does not apply to discrete variables, since the Gaussian distribution has nonzero probabilities for non-integer x values, whereas the probability of pulling a non-integer out of a distribution of integers is zero. Typically, you would use a Gaussian distribution as an approximation for a binomial distribution, where you set a=1 and set P(k) = G(x=k, mu, sigma)*a.

# For sufficiently large n, a binomial distribution and a Gaussian will appear similar according to

# B(k, p, n) =  G(x=k, mu=p*n, sigma=sqrt(p*(1-p)*n)).
# If you wish to fit a Gaussian distribution, you can use the standard scipy function scipy.stats.norm.fit. Such fit functions are not offered for the discrete distributions such as the binomial. You can use the function scipy.optimize.curve_fit to fit non-integer parameters such as the p parameter of the binomial distribution. In order to find the optimal integer n value, you need to vary n, fit p for each n, and pick the n, p combination with the best fit.

# In the implementation below, I estimate n and p from the relation with the mean and sigma value above and search around that value. The search could be made smarter, but for the small test datasets that I used, it's fast enough. Moreover, it helps illustrate a point; more on that later. I have provided a function fit_binom, which takes a histogram with actual counts, and a function fit_samples, which can take a column of numbers from your dataframe.

# In principle, the most best fit will be obtained if you set weighted=True. However, the question asks for the minimum sum of squared errors (SSE) as a metric; then, you can set weighted=False.

# It turns out that it is difficult to fit a binomial distribution unless you have a lot of data. Here are tests with realistic (random-generated) data for n, p combinations (10, 0.2), (10, 0.3), (10, 0.8), and (1, 0.3), for various numbers of samples. The plots also show how the weighted SSE changes with n.

# Typically, with 500 samples, you get a fit that looks OK by eye, but which does not recover the actual n and p values correctly, although the product n*p is quite accurate. In those cases, the SSE curve has a broad minimum, which is a giveaway that there are several reasonable fits.

# The code above can be adapted for different discrete distributions. In that case, you need to figure out reasonable initial estimates for the fit parameters. For example: Poisson: the mean is the only parameter (use the reduced chi2 or SSE to judge whether it's a good fit).

# If you want to fit a combination of m input columns to a (m+1)-dimensional multinomial , you can do a binomial fit on each input column and store the fit results in arrays nn and pp (each an array with shape (m,)). Transform these into an initial estimate for a multinomial:


# n_est = int(nn.mean()+0.5)
# pp_est = pp*nn/n_est
# pp_est = np.append(pp_est, 1-pp_est.sum())

# If the individual values in the nn array vary a lot, or if the last element of pp_est is negative, then it's probably not a multinomial.

# You want to compare the residuals of multiple models; be aware that a model that has more fit parameters will tend to produce lower residuals, but this does not necessarily mean that the model is better.

# Note: this answer underwent a large revision.



"""Binomial fit routines.

Author: Han-Kwang Nienhuys (2020); copying: CC-BY-SA.
https://stackoverflow.com/a/62365555/6228891 
"""

import numpy as np
from scipy.stats import binom, poisson
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class BinomPMF:
    """Wrapper so that integer parameters don't occur as function arguments."""
    def __init__(self, n):
        self.n = n
    def __call__(self, ks, p):
        return binom(self.n, p).pmf(ks)

def fit_binom(hist, plot=True, weighted=True, f=1.5, verbose=False):
    """Fit histogram to binomial distribution.
    
    Parameters:

    - hist: histogram as int array with counts, array index as bin.
    - plot: whether to plot
    - weighted: whether to fit assuming Poisson statistics in each bin.
      (Recommended: True).
    - f: try to fit n in range n0/f to n0*f where n0 is the initial estimate.
      Must be >= 1.
    - verbose: whether to print messages.
    
    Return: 
        
    - histf: fitted histogram as int array, same length as hist.
    - n: binomial n value (int)
    - p: binomial p value (float)
    - rchi2: reduced chi-squared. This number should be around 1.
      Large values indicate a bad fit; small values indicate
      "too good to be true" data.
    """ 
   
    hist = np.array(hist, dtype=int).ravel() # force 1D int array
    pmf = hist/hist.sum() # probability mass function
    nk = len(hist)
    if weighted:
        sigmas = np.sqrt(hist+0.25)/hist.sum()
    else:
        sigmas = np.full(nk, 1/np.sqrt(nk*hist.sum()))
    ks = np.arange(nk)
    mean = (pmf*ks).sum()
    variance = ((ks-mean)**2 * pmf).sum()
    
    # initial estimate for p and search range for n
    nest = max(1, int(mean**2 /(mean-variance) + 0.5))
    nmin = max(1, int(np.floor(nest/f)))
    nmax = max(nmin, int(np.ceil(nest*f)))
    nvals = np.arange(nmin, nmax+1)
    num_n = nmax-nmin+1
    verbose and print(f'Initial estimate: n={nest}, p={mean/nest:.3g}')

    # store fit results for each n
    pvals, sses = np.zeros(num_n), np.zeros(num_n)
    for n in nvals:
        # fit and plot
        p_guess = max(0, min(1, mean/n))
        fitparams, _ = curve_fit(
            BinomPMF(n), ks, pmf, p0=p_guess, bounds=[0., 1.],
            sigma=sigmas, absolute_sigma=True)
        p = fitparams[0]
        sse = (((pmf - BinomPMF(n)(ks, p))/sigmas)**2).sum()
        verbose and print(f'  Trying n={n} -> p={p:.3g} (initial: {p_guess:.3g}),'
                          f' sse={sse:.3g}')
        pvals[n-nmin] = p
        sses[n-nmin] = sse
    n_fit = np.argmin(sses) + nmin
    p_fit = pvals[n_fit-nmin]
    sse = sses[n_fit-nmin]    
    chi2r = sse/(nk-2) if nk > 2 else np.nan
    if verbose:
        print(f'  Found n={n_fit}, p={p_fit:.6g} sse={sse:.3g},'
              f' reduced chi^2={chi2r:.3g}')
    histf = BinomPMF(n_fit)(ks, p_fit) * hist.sum()

    if plot:    
        fig, ax = plt.subplots(2, 1, figsize=(4,4))
        ax[0].plot(ks, hist, 'ro', label='input data')
        ax[0].step(ks, histf, 'b', where='mid', label=f'fit: n={n_fit}, p={p_fit:.3f}')
        ax[0].set_xlabel('k')
        ax[0].axhline(0, color='k')
        ax[0].set_ylabel('Counts')
        ax[0].legend()
        
        ax[1].set_xlabel('n')
        ax[1].set_ylabel('sse')
        plotfunc = ax[1].semilogy if sses.max()>20*sses.min()>0 else ax[1].plot
        plotfunc(nvals, sses, 'k-', label='SSE over n scan')
        ax[1].legend()
        fig.show()
        
    return histf, n_fit, p_fit, chi2r

def fit_binom_samples(samples, f=1.5, weighted=True, verbose=False):
    """Convert array of samples (nonnegative ints) to histogram and fit.
    
    See fit_binom() for more explanation.
    """
    
    samples = np.array(samples, dtype=int)
    kmax = samples.max()
    hist, _ = np.histogram(samples, np.arange(kmax+2)-0.5)
    return fit_binom(hist, f=f, weighted=weighted, verbose=verbose) 

def test_case(n, p, nsamp, weighted=True, f=1.5):
    """Run test with n, p values; nsamp=number of samples."""
    
    print(f'TEST CASE: n={n}, p={p}, nsamp={nsamp}')
    ks = np.arange(n+1) # bins
    pmf = BinomPMF(n)(ks, p)
    hist = poisson.rvs(pmf*nsamp)
    fit_binom(hist, weighted=weighted, f=f, verbose=True)

if __name__ == '__main__':
    plt.close('all')
    np.random.seed(1)
    weighted = True
    test_case(10, 0.2, 500, f=2.5, weighted=weighted)
    test_case(10, 0.3, 500, weighted=weighted)
    test_case(10, 0.8, 10000, weighted)
    test_case(1, 0.3, 100, weighted) # equivalent to Bernoulli distribution
    fit_binom_samples(binom(15, 0.5).rvs(100), weighted=weighted)
    