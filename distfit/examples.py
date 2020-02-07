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

# %% Find best fit distribution 
X = np.random.beta(5, 8, [100,100])
y = [-1,-0.8,-0.6,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.5]

model = dist.fit(X)
dist.plot(model)
dist.plot_summary(model)

model = dist.proba_parametric(y, model=model)
dist.plot(model)

# %%
model = dist.fit(X)
out = dist.proba_parametric(y, model=model)
dist.plot(out)

# %%
out = dist.proba_emperical(y, X)
dist.plot(out)

# %%
X = np.random.beta(5, 8, 1000)

model = dist.proba_parametric(y, X)
dist.plot(model)

# %%
model = dist.proba_parametric(y, X, bound='up')
dist.plot(model)

model = dist.proba_parametric(y, X, bound='down')
dist.plot(model)

model = dist.proba_emperical(y, X)
dist.plot(model)

# %% Find best fit distribution ==============================
y = np.random.beta(5, 8, 10000)
model = dist.fit(y)
dist.plot(model)

y = np.random.normal(5, 8, [100,100])
model = dist.fit(y, distribution='auto_small')
dist.plot(model)

y = np.random.normal(5, 8, [100,100])
model = dist.fit(y, distribution='norm')
dist.plot(model)

# y = np.random.normal(5, 8, 10000)
# model = dist.fit(y, bins=50, distribution='auto_full')
# dist.plot(model)
