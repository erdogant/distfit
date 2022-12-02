import numpy as np
import distfit
# print(distfit.__version__)
# print(dir(distfit))


# %% K distributions as input
import scipy.stats as st
from distfit import distfit
X = np.random.normal(0, 2, 1000)
y = [-8, -6, 0, 1, 2, 3, 4, 5, 6]
dist = distfit(stats='ks', distr=['k','t','expon', 't', 'gamma', 'lognorm'])
results = dist.fit_transform(X)

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
results = dist.predict([0, 1, 10, 11, 12])
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

X = np.random.normal(10, 3, 2000)
y = [3,4,5,6,10,11,12,18,20]

# Initialize
# dist = distfit(method='percentile', alpha=0.05, todf=False)
# dist = distfit(method='quantile', alpha=0.05, todf=False)
dist = distfit(method='parametric', alpha=0.05, todf=False)
dist.fit_transform(X)
dist.plot()

# Make prediction
dist.predict(y)
dist.plot()
dist.plot_summary()

# from tabulate import tabulate
# print(tabulate(dist.results['df'], tablefmt="grid", headers="keys"))

# %%
