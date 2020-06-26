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

dist = distfit()
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
dist = distfit(method='quantile', alpha=0.05)
dist.fit_transform(X)
dist.plot()

# Make prediction
dist.predict(y)
dist.plot()
dist.plot_summary()

