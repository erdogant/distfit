# --------------------------------------------------
# Name        : examples.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/distfit
# Licence     : MIT
# --------------------------------------------------

import numpy as np
import distfit as dist
print(dist.__version__)


# %% Import class
from distfit import dist
dir(dist)


# %%
# Initialize
X = np.random.beta(5, 8, [100,100])
y = [-1,-0.8,-0.6,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.5]

model = dist()

# Fit and transform
model.fit()
model.transform(X)
model.plot()
model.predict(y)
model.plot()

# for Fit and transform in one go
model = dist()
model.fit_transform(X)
model.plot()
model.predict(y)
model.plot()

# Make plots
model.plot_summary()

# Show some results
model.y_proba
model.y_pred
model.df


# %%
X = np.random.normal(0, 2, 1000)
y = [-8,-6,0,1,2,3,4,5,6]

model = dist(method='emperical')
model.fit_transform(X)
model.plot()

# Make prediction
model.predict(y)
model.plot()


X = np.random.normal(5, 8, [100,100])
y = [-35, -10, 0, 10, 15, 35]

model = dist(method='emperical')
model.fit_transform(X)
model.predict(y)

model.plot()

model.y_proba
model.y_pred
model.df

# %%
X = np.random.beta(5, 8, 1000)

model = dist()
model.fit_transform(X)
model.plot()

# %% Find distribution parameters
X = np.random.normal(0, 2, 5000)
model = dist()
model.fit_transform(X)
model.plot()

X = np.random.normal(10, 1, 5000)
model = dist()
model.fit_transform(X)
model.plot()

X = np.random.normal(10, 5, 5000)
model = dist()
model.fit_transform(X)
model.plot()

# %%
X = np.random.normal(0, 2, 1000)
y = [-8,-6,0,1,2,3,4,5,6]

model = dist()
model.fit_transform(X)
model.predict(y)
model.plot()
model.df

model = dist(bound='up')
model.fit_transform(X)
model.predict(y)
model.plot()
model.df

model = dist(bound='down')
model.fit_transform(X)
model.predict(y)
model.plot()
model.df

# %% Find best fit distribution ==============================
X = np.random.normal(0, 2, 1000)
y = [-8,-6,0,1,2,3,4,5,6]

model = dist()
model.fit_transform(X)
model.plot()

model = dist(distribution='auto_small')
model.fit_transform(X)
model.plot()

model = dist(distribution='auto_full')
model.fit_transform(X)
model.plot()
