import numpy as np
import distfit as dist
print(dist.__version__)

# %% Find best fit distribution 
X = np.random.normal(0, 2, 1000)
y = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]

model = dist.fit(X, bins=100)
dist.plot(model)
dist.plot_summary(model)

model = dist.proba_parametric(X)
dist.plot(model)

# %%
model = dist.fit(X)
out = dist.proba_parametric(y, X, model=model)
dist.plot(out)

# %%
out = dist.proba_parametric(y, X)
dist.plot(out)

# %%

model = dist.proba_parametric(y, X)
dist.plot(model)

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
