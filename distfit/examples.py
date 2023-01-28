import numpy as np
from scipy.stats import binom, poisson
import matplotlib.pyplot as plt

import distfit
# print(distfit.__version__)
# print(dir(distfit))

# %%
from distfit import distfit
dist = distfit()
d = dist.get_distributions('popular')

X = np.random.normal(0, 1, 1000)
bins, density = dist.density(X)
plt.figure(); plt.plot(bins, density)

# %% Figure 1

# Load library
from distfit import distfit

# Random Exponential data
X = np.random.poisson(10, 10000)
X = np.random.uniform(0, 1000, 10000)
# initialize with uniform distribution
dist = distfit(distr='uniform')
# Fit and plot
results = dist.fit_transform(X, verbose='warning')
dist.plot(grid=False, cii_properties=None);

# Random exponential data
X = np.random.exponential(0.5, 10000)
# initialize with exponential distribution
dist = distfit(distr='expon')
# Fit and plot
results = dist.fit_transform(X, verbose='debug')
dist.plot(grid=False, cii_properties=None, verbose=10);

# Random normal distribution
X = np.random.normal(0, 2, 10000)
dist = distfit(distr='norm')
# Fit and plot
results = dist.fit_transform(X)
dist.plot(figsize=(15, 12), grid=False)

# Random bimodal distribution
X1 = list(np.random.normal(10, 3, 10000))
X2 = list(np.random.normal(0, 2, 2000))
X = np.array(X1+X2)
dist = distfit()
# Fit and plot
results = dist.fit_transform(X)
dist.plot(figsize=(15, 12), grid=False, cii_properties=None, pdf_properties=None)

# %% Figure 2
# Random normal distribution
X = np.random.normal(2, 4, 10000)
y = [-8, -2, 1, 3, 5, 15]
dist = distfit(distr='norm')
# dist = distfit(method='quantile')
# Fit and plot
dist.fit_transform(X);
dist.model

dist.predict(y);
dist.plot(figsize=(15, 12), grid=True);
dist.plot_summary()

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
dist = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
dist = distfit(method='discrete')
# Fit and plot
model = dist.fit_transform(X, verbose=3)
dist.plot(figsize=(15, 12), grid=True)

# %% Quantile approach
from distfit import distfit
import numpy as np

X1 = list(np.random.normal(10, 3, 10000))
X2 = list(np.random.normal(0, 2, 2000))
X = np.array(X1+X2)
y = [3,4,5,6,10,11,12,18,20]

# Initialize
# dist = distfit(method='percentile', alpha=0.05, todf=False)
# dist = distfit(method='quantile', alpha=0.05, todf=False)
dist = distfit(method='parametric', alpha=0.05, todf=False)
dist.fit_transform(X);
dist.plot(figsize=(15, 12), cii_properties=None, pdf_properties=None, grid=False)

# Make prediction
dist.predict(y);
dist.plot();
dist.plot_summary();


# %% Multiple distributions as input
from distfit import distfit
X = np.random.normal(0, 2, 10000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dist = distfit(stats='RSS', distr=['norm','expon'])
results = dist.fit_transform(X)
dist.plot(cii_properties={'size': 50})

results = dist.predict(y, alpha=0.01)
dist.plot(cii_properties={'size': 20, 'marker': 'x', 'linewidth':2})

print(dist.model)


from distfit import distfit
X = binom(8, 0.5).rvs(1000)
dist = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
model = dist.fit_transform(X, verbose=3)
dist.plot(figsize=(15, 12), grid=True)


from distfit import distfit
X = np.random.uniform(0, 1000, 10000)
dist = distfit(bound=None, distr='uniform')
results = dist.fit_transform(X)
dist.plot(figsize=(15, 12), grid=False)

from distfit import distfit
X = np.random.exponential(0.5, 10000)
dist = distfit(bound=None, distr='expon')
results = dist.fit_transform(X)
dist.plot(figsize=(15, 12), grid=False)
# dist.plot_summary()

from distfit import distfit
X = np.random.normal(0, 2, 10000)
dist = distfit(bound=None, distr='norm')
results = dist.fit_transform(X)
dist.plot(figsize=(15, 12), grid=False)

dist.plot(bar_properties={'color': 'r', 'label': None}, pdf_properties={'color': 'k'}, emp_properties={'color': 'k', 'linewidth': 3})
dist.plot(bar_properties=None, pdf_properties=None)
dist.plot(bar_properties={}, pdf_properties=None, emp_properties={})
dist.plot(bar_properties={}, pdf_properties={}, emp_properties=None)


# %% K distributions as input
import scipy.stats as st
from distfit import distfit
X = np.random.normal(0, 2, 1000)
dist = distfit(stats='ks', distr=['k','t','expon', 'gamma', 'lognorm'], bins=50)
results = dist.fit_transform(X, verbose=0)

dist.plot()
dist.plot_summary()

# %% Multiple distributions as input
import scipy.stats as st
from distfit import distfit
X = np.random.normal(0, 2, 1000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dist = distfit(stats='ks', distr=['expon', 't', 'gamma', 'lognorm'])
# dist = distfit(stats='ks', distr=['lognorm'])
results = dist.fit_transform(X)

# dist.plot()
# dist.plot_summary()

# results = dist.predict(y, alpha=0.01)

print(dist.model)
# dist_t = st.t(dist.model['arg'], loc=dist.model['loc'], scale=dist.model['scale'])
# dist_t = st.t(dist.model['params'])

# dist.predict(y)['P']
# dist_t.cdf(y)
# dist.model['model'].cdf(y)

# fit dist to data
params = st.t.fit(X)
# Separate parts of parameters
arg = params[:-2]
loc = params[-2]
scale = params[-1]

params==dist.model['params']

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
                dist = distfit(todf=True, stats=stat, bins=bs)
                dist.fit_transform(X)
                r = np.where(dist.summary['distr']=='pareto')[0][0]
                ranking.append([r, b, s, bs, stat])

df = pd.DataFrame(ranking, columns=['rank','b','sample size', 'bins', 'stats'])

np.sum(df['rank']==0) / df.shape[0]
np.sum(df['rank']<=1) / df.shape[0]
np.sum(df['rank']<=2) / df.shape[0]
np.sum(df['rank']<=3) / df.shape[0]

# Other distr. have better scores under these conditions
df.loc[df['rank']>=3, :]

dist.plot_summary()
# results['model']

# %% Multiple distributions as input
from distfit import distfit
X = np.random.normal(0, 2, 10000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dist = distfit(stats='RSS', distr=['norm','expon'])
results = dist.fit_transform(X)
dist.plot()

results = dist.predict(y, alpha=0.01)
dist.plot()

print(dist.model)

# %% Discrete example
from distfit import distfit
from scipy.stats import binom
# Generate random numbers
X = binom(8, 0.5).rvs(1000)

dist = distfit(method='discrete', f=1.5, weighted=True, stats='wasserstein')
model = dist.fit_transform(X, verbose=3)
dist.plot()

# Make prediction
results = dist.predict([0, 1, 3, 4, 10, 11, 12])
dist.plot()

# Generate samples
Xgen = dist.generate(n=1000)
dist.fit_transform(Xgen)
results = dist.predict([0, 1, 10, 11, 12])
dist.plot()

# %%
from distfit import distfit
X = np.random.normal(0, 2, 10000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dist = distfit(stats='RSS', distr='full')
# dist = distfit(stats='wasserstein')
# dist = distfit(stats='energy')
# dist = distfit(stats='ks')

# Fit
dist.fit_transform(X)
dist.predict(y)

dist.plot_summary()
dist.plot()

# Generate samples
Xgen = dist.generate(n=10000)
dist.fit_transform(Xgen)

# Plot
dist.plot_summary()
dist.plot()

# %%
from distfit import distfit
X = np.random.normal(0, 2, 5000)
y = [-8,-6,0,1,2,3,4,5,6]
dist = distfit(distr='loggamma')
dist.fit_transform(X)
dist.plot()

# %%
from distfit import distfit
X = np.random.normal(0, 2, 5000)
y = [-8,-6,0,1,2,3,4,5,6]

dist = distfit(distr='popular', todf=False)
model = dist.fit_transform(X)
dist.plot()

dist = distfit(distr='popular', todf=True)
dist.fit_transform(X)
dist.plot()

# Make prediction
results = dist.predict(y)

# plot
dist.plot()
dist.plot_summary()

# Save
dist.save(filepath='c:\\temp\\model.pkl', overwrite=True)
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

#%%
# Initialize model
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
samples = np.arange(250, 20000, 250)
smooth_window=[None,2,4,6,8,10]
plt.figure(figsize=(15,10))

for smooth in tqdm(smooth_window):
    dist = distfit(distr='norm', smooth=smooth)
    # Estimate paramters for the number of samples
    out = []
    for s in samples:
        X = np.random.normal(0, 2, s)
        dist.fit_transform(X, verbose=0)
        # out.append([dist.model['RSS'], dist.model['name'], np.where(dist.summary['distr']=='norm')[0][0], s])
        out.append([dist.model['scale'], dist.model['name'], s])

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
    dist = distfit(distr='uniform', smooth=smooth)
    # dist = distfit(smooth=smooth)
    # Estimate paramters for the number of samples
    out = []
    for s in samples:
        X = np.random.randint(0, 100, s)
        dist.fit_transform(X, verbose=0)
        # out.append([dist.model['RSS'], dist.model['name'], np.where(dist.summary['distr']=='uniform')[0][0], s])
        out.append([dist.model['RSS'], dist.model['name'], s])

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

dist = distfit(stats='wasserstein')
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
# dist = distfit(method='percentile', alpha=0.05, todf=False)
dist = distfit(method='quantile', alpha=0.05, todf=False)
# dist = distfit(method='parametric', alpha=0.05, todf=False)
dist.fit_transform(X)
dist.plot()

# Make prediction
dist.predict(y)
dist.plot()
dist.plot_summary()

# from tabulate import tabulate
# print(tabulate(dist.results['df'], tablefmt="grid", headers="keys"))

# %%
