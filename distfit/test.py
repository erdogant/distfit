import numpy as np
import distfit as dist
print(dist.__version__)

# %% Find best fit distribution 
data_random = np.random.normal(0, 2, 1000)
data = [-14,-8,-6,0,1,2,3,4,5,6,7,8,9,10,11,15]

model = dist.proba_parametric(data_random)
dist.plot(model)

# %%
model = dist.fit(data_random)
out = dist.proba_parametric(data, data_random, model=model)
dist.plot(out)

# %%
out = dist.proba_parametric(data, data_random)
dist.plot(out)

# %%

model = dist.proba_parametric(data, data_random)
dist.plot(model)

model = dist.proba_parametric(data, data_random, bound='up')
dist.plot(model)

model = dist.proba_parametric(data, data_random, bound='low')
dist.plot(model)

model = dist.proba_emperical(data, data_random)
dist.plot(model)

model = dist.proba_emperical(data)
dist.plot(model)


# %% Find best fit distribution ==============================
data = np.random.beta(5, 8, 10000)
model = dist.fit(data)
dist.plot(model)

data = np.random.normal(5, 8, [100,100])
model = dist.fit(data, distribution='auto_small')
dist.plot(model)

data = np.random.normal(5, 8, [100,100])
model = dist.fit(data, distribution='norm')
dist.plot(model)

# data = np.random.normal(5, 8, 10000)
# model = dist.fit(data, bins=50, distribution='auto_full')
# dist.plot(model)
