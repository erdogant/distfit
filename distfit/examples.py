import numpy as np
import matplotlib.pyplot as plt
from distfit import distfit
import time
from tqdm import tqdm
import colourmap

X = np.random.normal(163, 10, 10000)

def run_prog_bootstrap(n_jobs: int, n_boots: int):
    dfit = distfit(distr='popular', n_boots=n_boots, n_jobs=n_jobs, verbose='warning')
    results = dfit.fit_transform(X)
    return results

def compute_performance(n_boots_list):
    # combinations = [(1, 1)] + [(i, 8 - i) for i in range(1, 8)]
    combinations = np.arange(1, 14)
    timings = {}

    for comb in tqdm(combinations):
        timings[(comb)] = []
        for n_boots in tqdm(n_boots_list, ):
            start_time = time.time()
            _ = run_prog_bootstrap(comb, n_boots)
            elapsed_time = time.time() - start_time
            timings[(comb)].append((n_boots, elapsed_time))

    return timings

def plot_performance(timings):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # Define colors for different n_jobs values
    colors = colourmap.fromlist(np.arange(1, 14), scheme='hex')[0]
    markers = ['o', '^', 's', 'd', 'x', 'v', 'h']  # Define markers for different n_jobs values

    for n_jobs, results in timings.items():
        x, y = zip(*results)
        label = f"n_jobs={n_jobs}"
        # Ensure indexing within range of available colors and markers
        idx = min(n_jobs - 1, len(colors) - 1)
        ax.plot(x, y, label=label, color=colors[idx], marker='o', linestyle='-')

    ax.set_xlabel('n_boots')
    ax.set_ylabel('Time (s)')
    ax.set_title('Performance Comparison')
    ax.legend()
    plt.grid(True)
    plt.show()


# Define the n_boots_list
n_boots_list = [0, 10, 50, 100]
# Compute performance
timings = compute_performance(n_boots_list)
# Plot performance
plot_performance(timings)



# %% Parrellel computing
import time
from distfit import distfit

X = np.random.normal(163, 10, 10000)

start_time = time.time()

# dfit = distfit(distr='full', n_boots=50, n_jobs=-1, verbose='info')
dfit = distfit(distr='popular', n_jobs=20, n_boots=50, verbose='info')
dfit.fit_transform(X)

elapsed_time = time.time() - start_time
print(elapsed_time)
# n_jobs=-1 > 79.834397315979
# n_jobs=1 >  64 sec
# n_jobs=2 >  58 sec

# %% Issue 45
# https://github.com/erdogant/distfit/issues/45

import numpy as np
from distfit import distfit

data = np.array([ 56.518556,  54.803739,  57.424846,  54.254221,  63.235301,
        55.815964,  56.557519,  56.789227,  55.710028,  55.348868,
        55.998148,  54.88984 ,  60.698556,  58.037249,  55.998148,
        56.196659,  54.07792 ,  54.324279,  55.279121,  55.85325 ,
        54.677967,  54.330469,  54.122291,  54.819475,  54.565095,
       117.236973,  54.512653,  56.638532,  53.162648,  54.602637,
        56.66363 ,  56.934138,  59.085959,  57.303842,  58.934084,
       183.203797, 110.220432,  57.52065 ,  54.509817, 129.639834,
        69.668429, 126.631612, 148.791635,  85.291877, 145.450409,
        58.601726,  83.397137,  62.084062,  54.81671 ,  59.890595,
        95.307584,  63.366694,  54.16292 , 151.382722,  58.215827,
       147.623722, 119.041469, 114.503229,  66.526126, 138.969765,
       135.064926, 146.308008, 139.331183, 125.15503 , 150.57275 ,
        59.308423,  58.144718,  57.447888, 149.112722, 142.705007,
       133.288753, 141.13678 , 147.519795, 140.110123, 152.173592,
       146.103209, 147.683985, 147.416646, 150.066857, 142.576063,
       144.83517 , 145.818179, 145.499275, 145.83333 , 145.298108,
       146.885261, 145.397002, 145.282957, 145.648398, 145.884035,
       146.648887])

marg_dists = ['gennorm',
              'genlogistic',
              'mielke',
              'johnsonsu',
              'burr',
              'johnsonsb',
              'loggamma',
              'norminvgauss']

dfit = distfit(distr=marg_dists)
dfit.fit_transform(data)
dfit.plot(bar_properties={'width':10})

# %% Issue xx

from distfit import distfit
import numpy as np
X = np.random.normal(0, 2, 10000)

dfit = distfit()
dfit.fit_transform(X)
dfit.plot(bar_properties={'width':0.1})



# %% Import libraries
import time
import numpy as np
from distfit import distfit

# Create random normal data with mean=2 and std=4
X = np.random.normal(2, 4, 10000)

# Alternatively limit the search for only a few theoretical distributions.
dfit = distfit(method='parametric', todf=True, n_jobs=-1)

# Fit model on input data X.
start = time.time()
dfit.fit_transform(X, n_boots=100)
end = time.time()
print(end - start)

dfit.summary[['name', 'score', 'loc', 'scale']]

# Bootstrapping
# dfit.bootstrap(X, n_boots=100)

# Print
dfit.plot_summary()
print(dfit.summary[['name', 'score', 'bootstrap_score', 'bootstrap_pass']])



# %% Issue 39
from distfit import distfit
import time
import numpy as np

dfit = distfit(todf=True, distr='t', n_jobs=1)
cost = np.random.normal(0, 2, 10000)

start = time.time()
result = dfit.fit_transform(cost, n_boots=10)
end = time.time()
print(end - start)



from distfit import distfit
import matplotlib.pyplot as plt

dfit = distfit(smooth=3, bound='up')
df = dfit.import_example(data='tips')
dfit.fit_transform(df['tip'], n_boots=100)

dfit.summary


# %% Issue 39
from distfit import distfit
import numpy as np

dfit = distfit(todf=True, distr='t')
cost = np.random.normal(0, 2, 10000)
result = dfit.fit_transform(cost, n_boots=100)
dfit.plot(chart='pdf')

# %%



# %%
import numpy as np
from distfit import distfit
import matplotlib.pyplot as plt
X = np.random.normal(0, 2, 10000)
y = [-8,-6,0,1,2,3,4,5,6]
dfit = distfit(alpha=0.01)
dfit.fit_transform(X)
fig, ax = dfit.plot(chart='pdf')
fig, ax = dfit.plot(chart='cdf', n_top=11)

plt.show()
# Or
fig.show()




# %%

from distfit import distfit
import matplotlib.pyplot as plt

dfit = distfit(smooth=3)
df = dfit.import_example(data='tips')
dfit.fit_transform(df['tip'])

dfit.summary = dfit.summary[dfit.summary['name'].isin(['lognorm', 'gamma', 'pareto'])]
dfit.plot(chart='pdf', n_top=5)


# %%
from distfit import distfit

dfit = distfit(smooth=3, bound='up')
df = dfit.import_example(data='tips')
dfit.lineplot(df['tip'], xlabel='Number', ylabel='Tip value', grid=True, line_properties={'marker':'.'}, projection=True)

dfit.fit_transform(df['tip'])
# dfit.fit_transform(df['tip'], n_boots=0)
dfit.lineplot(df['tip'], xlabel='Number', ylabel='Tip value', grid=True, line_properties={'marker':'.'}, projection=True, ylim=[0, 11])
# dfit.lineplot(df['tip'], xlabel='Number', ylabel='Tip value', grid=True, line_properties={'marker':'.'}, projection=False)
# dfit.plot()

X = dfit.generate(100)
# Ploy the data
dfit.lineplot(X, xlabel='Number', ylabel='Tip value', grid=True, ylim=[0, 11])


# %%
from distfit import distfit
import time
import pandas as pd

start_time = time.time()
# normal_distributions = ["halfgennorm", "lognorm", "powerlognorm", "gennorm", "norm", "truncnorm", "exponnorm", "powernorm", "foldnorm",  "halfnorm"]
normal_distributions = ["halfgennorm", "lognorm"]

X = pd.read_pickle(r"C:\Users\playground\Downloads\repro\repro_data")
dfit = distfit(distr=normal_distributions, n_boots=1, n_jobs=1)
dfit.fit_transform(X)
time_spent = time.time() - start_time
print(time_spent)

# %%
from distfit import distfit
import pandas as pd

normal_distributions = ["halfgennorm", "lognorm", "powerlognorm",
                        "gennorm", "norm", "truncnorm", "exponnorm",
                        "powernorm", "foldnorm",  "halfnorm"]

values = pd.read_pickle(r"C:\Users\playground\Downloads\repro\repro_data")
dist_fitter = distfit(distr=normal_distributions, n_boots=10)
dist_fitter.fit_transform(values)
dist_fitter.plot_summary()
dist_fitter.plot()
dist_fitter.bootstrap(values, n_boots=10, update_model=False)

# %%
import matplotlib.pyplot as plt
from distfit import distfit
import seaborn as sns
import numpy as np
import scipy.stats as st
np.random.seed(4)

loc = 5
scale=10
sample_dist = st.lognorm.rvs(3, loc=loc, scale=np.exp(scale), size=10000)
# shape, loc, scale = st.lognorm.fit(sample_dist, floc=0)
# print(shape, loc, scale)
# print(np.log(scale), shape) # mu and sigma

dfit = distfit('parametric', todf=True, distr=["lognorm"])
dfit.fit_transform(sample_dist)

print('Estimated loc: %g, input loc: %g' %(dfit.model['loc'], loc))
print('Estimated mu or scale: %g, input scale: %g' %(np.log(dfit.model['scale']), scale))

# mu
np.mean(np.log(sample_dist))


mu=13.8
loc=47.55
x_sim = np.random.normal(loc=loc,scale=np.exp(mu), size = 10000)
x_sim = np.append([*filter(lambda x: x<=80, x_sim)],np.random.normal(loc=90,scale=10, size = 50))
x_sim = np.array([*filter(lambda x: x >=0,x_sim)])

# shape, loc, scale = stats.lognorm.fit(sample, floc=0) # hold location to 0 while fitting
# np.mean(x_sim)
# np.std(np.log(x_sim))

dfit = distfit(todf=True, distr=["lognorm"])
dfit.fit_transform(x_sim)
dfit.bootstrap(x_sim, n_boots=1)

# print('Estimated loc: %g, input loc: %g' %(dfit.model['loc'], loc))
print('Estimated mu or scale: %g, input scale: %g' %(np.log(dfit.model['scale']), mu))
dfit.plot()

np.mean(np.log(x_sim))
np.std(np.log(x_sim))


# sns.histplot(x,ax=ax[0])
fig, ax = plt.subplots(1,3, figsize=(20,8))
dfit.plot("PDF",n_top=3,fontsize=11, pdf_properties=None, cii_properties=None, emp_properties=None, ax=ax[0], bar_properties={'edgecolor': '#000000'})
dfit.plot("PDF",n_top=3,fontsize=11,ax=ax[1])
dfit.plot("CDF",n_top=3,fontsize=11,ax=ax[2])
plt.show()

# %%
from distfit import distfit
import matplotlib.pyplot as plt

dfit = distfit(smooth=3, bound='up')
df = dfit.import_example(data='tips')
dfit.fit_transform(df['tip'], n_boots=10)
# dfit.fit_transform(df['tip'], n_boots=0)
dfit.lineplot(df['tip'], xlabel='Number', ylabel='Tip value', grid=True)


# Plot PDF/CDF
fig, ax = plt.subplots(1,2, figsize=(25, 10))
dfit.plot(chart='pdf', n_top=5, ax=ax[0])
dfit.plot(chart='cdf', n_top=5,ax=ax[1])
# Show plot
plt.show()

fig, ax = dfit.plot()
dfit.plot(chart='cdf', ax=ax)

dfit.plot_summary()

X = dfit.generate(100)
# Ploy the data
dfit.lineplot(X, xlabel='Number', ylabel='Tip value', grid=True)
plt.figure();plt.plot(X)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from distfit import distfit
import scipy.stats as st

fig, ax = plt.subplots(3,4, figsize=(25,20))
distr = ['norm', 'expon', 'uniform', 't', 'beta', 'gamma', 'pareto', 'gengamma', 'lognorm', 'dweibull', 'cauchy', 'f']
# distr = ['norm', 'expon', 'uniform', 't', 'beta', 'gamma', 'pareto', 'gengamma', 'lognorm', 'dweibull']
# for i, row in enumerate(ax):
#     dfit = distfit(distr=distr[i])
#     # Fit
#     X = dfit.generate(n=1000)
#     # fig, ax = dfit.plot(bar_properties=None, emp_properties=None, cii_properties=None)
#     sns.histplot(data=X, kde=True, stat="density", ax=row)
#     # Results

# for i in range(3):
#     for j in range(4):
#         d = distr[i*4+j]
#         dfit = distfit(distr=d)
#         # Fit
#         X = dfit.generate(n=1000)
#         sns.histplot(data=X, kde=True, stat="density", ax=ax[i][j])
#         ax[i][j].set_title(distr[i*4+j], fontsize=16)
#         # Results

fig, ax = plt.subplots(3,4, figsize=(25,20))
size=5000
samples = {}
for dist in distr:
    if dist == 'norm':
        X = st.norm.rvs(size=size)
        i, j = 0, 0
    elif dist == 'expon':
        X = st.expon.rvs(size=size)
        i, j = 0, 1
    elif dist == 'uniform':
        X = st.uniform.rvs(size=size)
        i, j = 0, 2
    elif dist == 'beta':
        X = st.beta.rvs(2, 5, size=size)
        i, j = 0, 3
    elif dist == 't':
        X = st.t.rvs(5, size=size)
        i, j = 1, 0
    elif dist == 'gamma':
        X = st.gamma.rvs(2, size=size)
        i, j = 1, 1
    elif dist == 'pareto':
        X = st.pareto.rvs(2, size=size)
        i, j = 1, 2
    elif dist == 'gengamma':
        X = st.gengamma.rvs(a=1.5, c=1, scale=2, size=size)
        i, j = 1, 3
    elif dist == 'lognorm':
        X = st.lognorm.rvs(s=1, loc=0, scale=1, size=size)
        i, j = 2, 0
    elif dist == 'dweibull':
        X = st.weibull_min.rvs(c=1.5, loc=0, scale=1, size=size)
        i, j = 2, 1
    elif dist == 'cauchy':
        X = st.cauchy.rvs(loc=0, scale=1, size=200)
        i, j = 2, 2
    elif dist == 'f':
        f_samples = st.f.rvs(dfn=3, dfd=5, loc=0, scale=1, size=200)
        i, j = 2, 3
    sns.histplot(data=X, kde=True, stat="density", ax=ax[i][j])
    ax[i][j].set_title(dist, fontsize=16)
    ax[i][j].grid(True)
 
# %%
import numpy as np
from scipy.stats import binom, poisson
import matplotlib.pyplot as plt

import distfit
# print(distfit.__version__)
# print(dir(distfit))

from distfit import distfit
X = np.random.normal(163, 10, 1000)
dfit = distfit()
dfit.fit_transform(X, n_boots=0)
dfit.plot_summary()


# %%

from distfit import distfit
X = np.random.normal(6, 1, 500)
dfit = distfit()
dfit.fit_transform(X)

X2 = np.random.normal(6, 0.9, 500)
X2 = np.hstack([[7]*50, X2])
dfit2 = distfit()
dfit2.fit_transform(X2)

fig, ax = plt.subplots(1,2, figsize=(25,10))
dfit.plot(title='without swear words', ax=ax[0])
dfit2.plot(title='with swear words', ax=ax[1])
fig

import scipy.stats as st
st.kstest(X, X2)

# %% lineplot
from distfit import distfit
X = np.random.normal(163, 10, 1000)
dfit = distfit(multtest='test')
dfit.fit_transform(X, n_boots=10)
dfit.plot_summary()
dfit.plot_summary(ylim=[0, 0.0002])

dfit.lineplot(X)

y = [135, 145, 150, 160, 180, 185, 200]
results = dfit.predict(y, multtest='holm')
dfit.lineplot(X)

# %% Import example

from distfit import distfit
dfit = distfit()
df = dfit.import_example(data='gas_spot_price')
dfit.lineplot(df, xlabel='Years', ylabel='Natural gas spot price', grid=True)
plt.show()



# %% CDF plot
from distfit import distfit

# Random Exponential data
# X = np.random.exponential(0.5, 10000)
# X = np.random.uniform(0, 1000, 10000)
X = np.random.normal(163, 10, 10000)
# Initialize with bootstrapping
dfit = distfit(n_boots=1000)
# Fit
results = dfit.fit_transform(X)
# Results
dfit.summary[['name', 'score', 'bootstrap_score', 'bootstrap_pass']]
# Plot
dfit.plot_summary()


out = dfit.bootstrap(X, n_boots=10, n_top=None)
dfit.plot_summary()

fig, ax = dfit.plot(chart='pdf', n_top=5, cmap='Set2');
dfit.plot(chart='cdf', n_top=10, cmap='Set2', ax=ax);
dfit.plot_cdf()


# %%
from distfit import distfit
# Set parameters for the test-case
n = 8
p = 0.5
# Generate 10000 samples of the distribution of (n, p)
X = binom(n, p).rvs(10000)

# Initialize distfit for discrete distribution for which the binomial distribution is used. 
dfit = distfit(method='discrete')
# Run distfit to and determine whether we can find the parameters from the data.
results = dfit.fit_transform(X)
# Get the model and best fitted parameters.

y = [0, 1, 10, 11, 12]

results = dfit.predict(y)
dfit.plot(chart='pdf')
dfit.plot(chart='pdf', pdf_properties=None)
dfit.plot(chart='cdf', n_top=5)
dfit.plot(chart='cdf', pdf_properties=None, n_top=2)
dfit.plot_cdf()

# %% QQ plot
from distfit import distfit
dfit = distfit(verbose=20)

# Random Exponential data
# X = np.random.exponential(0.5, 10000)
# X = np.random.uniform(0, 1000, 10000)
X = np.random.normal(0, 1, 10000)
dfit = distfit(distr='popular')
# Fit
dfit.fit_transform(X)
# QQplot
dfit.qqplot(X)
dfit.qqplot(X, n_top=11, cmap='Set1')

dfit.plot(chart='pdf')
dfit.plot(chart='pdf', pdf_properties=None)
dfit.plot(chart='pdf', pdf_properties=None, n_top=10)
dfit.plot(chart='cdf', n_top=10)
dfit.plot(chart='cdf', pdf_properties=None, n_top=10)


# fig, ax = dfit.plot(chart='cdf', n_top=10);
# dfit.plot(chart='pdf', n_top=10, fig=fig, ax=ax);
# dfit.qqplot(X, n_top=10, fig=fig, ax=ax);

# %% CDF plot
from distfit import distfit
dfit = distfit(verbose=20)

# Random Exponential data
X = np.random.exponential(0.5, 10000)
# X = np.random.uniform(0, 1000, 10000)
# X = np.random.normal(0, 1, 1000)
dfit = distfit(distr='popular')
# Fit and plot
dfit.fit_transform(X)
# dfit.plot_cdf(n_top=10);
fig, ax = dfit.plot(chart='pdf', n_top=5, cmap='Set2');
dfit.plot(chart='cdf', n_top=10, cmap='Set2', ax=ax);
# dfit.plot_cdf()
dfit.plot_summary(n_top=10);

import scipy.stats as stats
import pylab 
# Calculate quantiles for a probability plot, and optionally show the plot.
# Generates a probability plot of sample data against the quantiles of a specified theoretical distribution.
# probplot optionally calculates a best-fit line for the data and plots the results using Matplotlib or a given plot function.
fig, ax = plt.subplots(figsize=(15,10))
out = stats.probplot(X, dist=dfit.model['name'], sparams=dfit.model['params'], plot=ax)
ax.grid(True)
    

# %%
from distfit import distfit
# Set parameters for the test-case
n = 8
p = 0.5
# Generate 10000 samples of the distribution of (n, p)
X = binom(n, p).rvs(10000)

# Initialize distfit for discrete distribution for which the binomial distribution is used. 
dfit = distfit(method='discrete')
# Run distfit to and determine whether we can find the parameters from the data.
results = dfit.fit_transform(X)
# Get the model and best fitted parameters.

y = [0, 1, 10, 11, 12]

results = dfit.predict(y)
dfit.plot(fontsize=14)

# %%
from distfit import distfit
dfit = distfit()
d = dfit.get_distributions('popular')

X = np.random.normal(0, 1, 1000)
bins, density = dfit.density(X)
plt.figure(); plt.plot(bins, density)

# %% Figure 1

# Load library
from distfit import distfit

# Random Exponential data
X = np.random.poisson(10, 10000)
X = np.random.uniform(0, 1000, 10000)
# initialize with uniform distribution
dfit = distfit(distr='uniform')
# Fit and plot
results = dfit.fit_transform(X, verbose='warning')
dfit.plot(grid=False, cii_properties=None);

# Random exponential data
X = np.random.exponential(0.5, 10000)
# initialize with exponential distribution
dfit = distfit(distr='expon')
# Fit and plot
results = dfit.fit_transform(X, verbose='debug')
dfit.plot(grid=False, cii_properties=None, verbose=10);

# Random normal distribution
X = np.random.normal(0, 2, 10000)
dfit = distfit(distr='norm')
# Fit and plot
results = dfit.fit_transform(X)
dfit.plot(figsize=(15, 12), grid=False)

# Random bimodal distribution
X1 = list(np.random.normal(10, 3, 10000))
X2 = list(np.random.normal(0, 2, 2000))
X = np.array(X1+X2)
dfit = distfit()
# Fit and plot
results = dfit.fit_transform(X)
dfit.plot(figsize=(15, 12), grid=False, cii_properties=None, pdf_properties=None)

# %% Figure 2
# Random normal distribution
X = np.random.normal(2, 4, 10000)
y = [-8, -2, 1, 3, 5, 15]
dfit = distfit(distr='norm')
# dfit = distfit(method='quantile')
# Fit and plot
dfit.fit_transform(X);
dfit.model

dfit.predict(y);
dfit.plot(figsize=(15, 12), grid=True);
dfit.plot_summary()

# Create random normal data with mean=2 and std=4
X = np.random.normal(2, 4, 10000)
# Load library
from distfit import distfit
# Initialize using the quantile or percentile approach.
model = distfit(method='quantile') # percentile
# Fit model on input data X and detect the best theoretical distribution.
model.fit_transform(X);
# Make prediction for new data points.
y = [-8, -2, 1, 3, 5, 15]
model.predict(y)
# Plot the results
model.plot()


# Random discrete data
X = binom(8, 0.5).rvs(1000)
dfit = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
dfit = distfit(method='discrete')
# Fit and plot
model = dfit.fit_transform(X, verbose=3)
dfit.plot(figsize=(15, 12), grid=True)

# %% Quantile approach
from distfit import distfit
import numpy as np

X1 = list(np.random.normal(10, 3, 10000))
X2 = list(np.random.normal(0, 2, 2000))
X = np.array(X1+X2)
y = [3,4,5,6,10,11,12,18,20]

# Initialize
# dfit = distfit(method='percentile', alpha=0.05, todf=False)
# dfit = distfit(method='quantile', alpha=0.05, todf=False)
dfit = distfit(method='parametric', alpha=0.05, todf=False)
dfit.fit_transform(X);
dfit.plot(figsize=(15, 12), cii_properties=None, pdf_properties=None, grid=False)

# Make prediction
dfit.predict(y);
dfit.plot();
dfit.plot_summary();


# %% Multiple distributions as input
from distfit import distfit
X = np.random.normal(0, 2, 10000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dfit = distfit(stats='RSS', distr=['norm','expon'])
results = dfit.fit_transform(X)
dfit.plot(cii_properties={'size': 50})

results = dfit.predict(y, alpha=0.01)
dfit.plot(cii_properties={'size': 20, 'marker': 'x', 'linewidth':2})

print(dfit.model)


from distfit import distfit
X = binom(8, 0.5).rvs(1000)
dfit = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
model = dfit.fit_transform(X, verbose=3)
dfit.plot(figsize=(15, 12), grid=True)


from distfit import distfit
X = np.random.uniform(0, 1000, 10000)
dfit = distfit(bound=None, distr='uniform')
results = dfit.fit_transform(X)
dfit.plot(figsize=(15, 12), grid=False)

from distfit import distfit
X = np.random.exponential(0.5, 10000)
dfit = distfit(bound=None, distr='expon')
results = dfit.fit_transform(X)
dfit.plot(figsize=(15, 12), grid=False)
# dfit.plot_summary()

from distfit import distfit
X = np.random.normal(0, 2, 10000)
dfit = distfit(bound=None, distr='norm')
results = dfit.fit_transform(X)
dfit.plot(figsize=(15, 12), grid=False)

dfit.plot(bar_properties={'color': 'r', 'label': None}, pdf_properties={'color': 'k'}, emp_properties={'color': 'k', 'linewidth': 3})
dfit.plot(bar_properties=None, pdf_properties=None)
dfit.plot(bar_properties={}, pdf_properties=None, emp_properties={})
dfit.plot(bar_properties={}, pdf_properties={}, emp_properties=None)


# %% K distributions as input
import scipy.stats as st
from distfit import distfit
X = np.random.normal(0, 2, 1000)
dfit = distfit(stats='ks', distr=['k','t','expon', 'gamma', 'lognorm'], bins=50)
results = dfit.fit_transform(X, verbose=0)

dfit.plot()
dfit.plot_summary(fontsize=10)

# %% Multiple distributions as input
import scipy.stats as st
from distfit import distfit
X = np.random.normal(0, 2, 1000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dfit = distfit(stats='ks', distr=['expon', 't', 'gamma', 'lognorm'])
# dfit = distfit(stats='ks', distr=['lognorm'])
results = dfit.fit_transform(X)

# dfit.plot()
# dfit.plot_summary()

# results = dfit.predict(y, alpha=0.01)

print(dfit.model)
# dist_t = st.t(dfit.model['arg'], loc=dfit.model['loc'], scale=dfit.model['scale'])
# dist_t = st.t(dfit.model['params'])

# dfit.predict(y)['P']
# dist_t.cdf(y)
# dfit.model['model'].cdf(y)

# fit dist to data
params = st.t.fit(X)
# Separate parts of parameters
arg = params[:-2]
loc = params[-2]
scale = params[-1]

params==dfit.model['params']

# Calculate fitted PDF and error with fit in distribution
# pdf = st.t.pdf(X, loc=loc, scale=scale, *arg)

# %% Multiple distributions as input
from distfit import distfit
import scipy.stats as st
import pandas as pd

ranking = []
b_pareto = [0.75, 1, 2, 3, 4, 5]
size = [100, 1000, 10000]
bins = [50, 100, 200]
stats = ['RSS', 'wasserstein']

for stat in stats:
    for bs in bins:
        for b in b_pareto:
            for s in size:
                X = st.pareto.rvs(b, size=s)
                dfit = distfit(todf=True, stats=stat, bins=bs)
                dfit.fit_transform(X)
                r = np.where(dfit.summary['name']=='pareto')[0][0]
                ranking.append([r, b, s, bs, stat])

df = pd.DataFrame(ranking, columns=['rank','b','sample size', 'bins', 'stats'])

np.sum(df['rank']==0) / df.shape[0]
np.sum(df['rank']<=1) / df.shape[0]
np.sum(df['rank']<=2) / df.shape[0]
np.sum(df['rank']<=3) / df.shape[0]

# Other distr. have better scores under these conditions
df.loc[df['rank']>=3, :]

dfit.plot_summary()
# results['model']

# %% Multiple distributions as input
from distfit import distfit
X = np.random.normal(0, 2, 10000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dfit = distfit(stats='RSS', distr=['norm','expon'])
results = dfit.fit_transform(X)
dfit.plot()

results = dfit.predict(y, alpha=0.01)
dfit.plot()

print(dfit.model)

# %% Discrete example
from distfit import distfit
from scipy.stats import binom
# Generate random numbers
X = binom(8, 0.5).rvs(1000)

dfit = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
model = dfit.fit_transform(X, verbose=3)
dfit.plot()

# Make prediction
results = dfit.predict([0, 1, 3, 4, 10, 11, 12])
dfit.plot()

# Generate samples
Xgen = dfit.generate(n=1000)
dfit.fit_transform(Xgen)
results = dfit.predict([0, 1, 10, 11, 12])
dfit.plot()

# %%
from distfit import distfit
X = np.random.normal(0, 2, 10000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dfit = distfit(stats='RSS', distr='full')
# dfit = distfit(stats='wasserstein')
# dfit = distfit(stats='energy')
# dfit = distfit(stats='ks')

# Fit
dfit.fit_transform(X)
dfit.predict(y)

dfit.plot_summary()
dfit.plot()

# Generate samples
Xgen = dfit.generate(n=10000)
dfit.fit_transform(Xgen)

# Plot
dfit.plot_summary()
dfit.plot()

# %%
from distfit import distfit
X = np.random.normal(0, 2, 5000)
y = [-8,-6,0,1,2,3,4,5,6]
dfit = distfit(distr='loggamma')
dfit.fit_transform(X)
dfit.plot()

# %%
from distfit import distfit
X = np.random.normal(0, 2, 5000)
y = [-8,-6,0,1,2,3,4,5,6]

dfit = distfit(distr='popular', todf=False)
model = dfit.fit_transform(X)
dfit.plot()

dfit = distfit(distr='popular', todf=True)
dfit.fit_transform(X)
dfit.plot()

# Make prediction
results = dfit.predict(y)

# plot
dfit.plot()
dfit.plot_summary()

# Save
dfit.save(filepath='c:\\temp\\model.pkl', overwrite=True)
# Load
dfit.load(filepath='c:\\temp\\model.pkl')

# Store entire object


# %%
X = np.random.normal(0, 2, 100)
model = distfit(smooth=10)
model.fit_transform(X)
model.plot()

# %%
# Create random data with varying number of samples

#%%
# Initialize model
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
samples = np.arange(250, 20000, 250)
smooth_window=[None,2,4,6,8,10]
plt.figure(figsize=(15,10))

for smooth in tqdm(smooth_window):
    dfit = distfit(distr='norm', smooth=smooth)
    # Estimate paramters for the number of samples
    out = []
    for s in samples:
        X = np.random.normal(0, 2, s)
        dfit.fit_transform(X, verbose=0)
        # out.append([dfit.model['RSS'], dfit.model['name'], np.where(dfit.summary['name']=='norm')[0][0], s])
        out.append([dfit.model['scale'], dfit.model['name'], s])

    df=pd.DataFrame(out, columns=['mu','name','samples'])
    ax=df['mu'].plot(grid=True, label='smooth: '+str(smooth) + ' - ' + str(df['mu'].mean()))

ax.set_xlabel('Nr.Samples')
ax.set_ylabel('mu')
ax.set_xticks(np.arange(0,len(samples)))
ax.set_xticklabels(samples.astype(str), rotation = 90)
# ax.set_ylim([0, 0.02])
# ax.set_ylim([1.9, 2.1])
ax.legend()

# ax=df['std'].plot(grid=True)
# ax.set_xlabel('Nr.Samples')
# ax.set_ylabel('std')
# ax.set_xticks(np.arange(0,len(samples)))
# ax.set_xticklabels(samples.astype(str))

#%%
# Initialize model
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
samples = np.arange(250, 20000, 250)
smooth_window=[None, 2,4,6,8,10]
plt.figure(figsize=(15,10))

for smooth in tqdm(smooth_window):
    dfit = distfit(distr='uniform', smooth=smooth)
    # dfit = distfit(smooth=smooth)
    # Estimate paramters for the number of samples
    out = []
    for s in samples:
        X = np.random.randint(0, 100, s)
        dfit.fit_transform(X, verbose=0)
        # out.append([dfit.model['RSS'], dfit.model['name'], np.where(dfit.summary['name']=='uniform')[0][0], s])
        out.append([dfit.model['RSS'], dfit.model['name'], s])

    df = pd.DataFrame(out, columns=['RSS','name','samples'])
    ax=df['RSS'].plot(grid=True, label='smooth: '+str(smooth) + ' - RSS: ' + str(df['RSS'].mean()))

ax.set_xlabel('Nr.Samples')
ax.set_ylabel('RSS')
ax.set_xticks(np.arange(0,len(samples)))
ax.set_xticklabels(samples.astype(str), rotation = 90)
ax.set_ylim([0, 0.0005])
ax.legend()


# %% Fit and transform
X = np.random.beta(5, 8, [100,100])
y = [-1,-0.8,-0.6,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.5]

dfit = distfit(stats='wasserstein')
dfit.fit()
dfit.transform(X)
dfit.plot()
dfit.predict(y)
dfit.plot()

dfit.plot_summary()

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
print(model.results['y_proba'])
print(model.results['y_pred'])
# print(model.results['df'])
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
model.results['y_proba']
model.results['y_pred']

model = distfit(todf=True)
model.fit_transform(X)
model.predict(y)
model.plot()

model.results['y_proba']
model.results['y_pred']
model.results['df'] # Only availble when using todf=True

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

model = distfit(todf=True)
model.fit_transform(X)
model.predict(y)
model.plot()

model.results['y_proba']
model.results['y_pred']
model.results['df']


model = distfit(bound='up', todf=True)
model.fit_transform(X)
model.predict(y)
model.plot()
model.results['df']

model = distfit(bound='down', todf=True)
model.fit_transform(X)
model.predict(y)
model.plot()
model.results['df']

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

X = np.random.normal(10, 3, 10000)
y = [3,4,5,6,10,11,12,18,20]

# Initialize
# dfit = distfit(method='percentile', alpha=0.05, todf=False)
dfit = distfit(method='quantile', alpha=0.05, todf=False)
# dfit = distfit(method='parametric', alpha=0.05, todf=False)
dfit.fit_transform(X)
dfit.plot()

# Make prediction
dfit.predict(y)
dfit.plot()
dfit.plot_summary()

# from tabulate import tabulate
# print(tabulate(dfit.results['df'], tablefmt="grid", headers="keys"))

# %%
