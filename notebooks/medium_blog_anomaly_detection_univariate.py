import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from distfit import distfit
from colourmap import colourmap

# Load the data into a pandas DataFrame
dfit = distfit()
df = dfit.import_example(data='gas_spot_price')

#             price
# date             
# 2023-02-07   2.35
# 2023-02-06   2.17
# 2023-02-03   2.40
# 2023-02-02   2.67
# 2023-02-01   2.65
#           ...
# 1997-01-13   4.00
# 1997-01-10   3.92
# 1997-01-09   3.61
# 1997-01-08   3.80
# 1997-01-07   3.82

# [6555 rows x 1 columns]

# %% Make plot
dfit.lineplot(df, xlabel='Years', ylabel='Natural gas spot price', grid=True)
plt.show()


# %% Make plot
# Get unique years
import matplotlib as mpl
uiyears = np.unique(df.index.year)
colors = colourmap.fromlist(uiyears, cmap='Set2', method='seaborn')[1]

# Create new figure
plt.figure(figsize=(35,15))
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Plot price actions
plt.plot(df.index, df['price'], color='black')


# Add vertical bands for each year
for year in uiyears:
    ax.axvspan(str(year)+'-01-01', str(year)+'-12-31', color=colors.get(year), alpha=0.2)

# Set propertie
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xticks(rotation=45)
plt.ylabel('Natural Gas Spot Price', fontsize=26)
plt.xlabel('Date', fontsize=26)
plt.grid(True)

# Show the plot
plt.show()

# %% Fit distribution
# Initialize
dfit = distfit(alpha=0.01, n_boots=100)
# Search for best theoretical fit on your empirical data
results = dfit.fit_transform(df['price'].values)

n_top=1

fig, ax = plt.subplots(1,2, figsize=(25, 10))
dfit.plot(chart='PDF', n_top=n_top, ax=ax[0])
dfit.plot(chart='CDF', n_top=n_top, ax=ax[1])

fig, ax = plt.subplots(1,2, figsize=(25, 10))
dfit.plot_summary(ax=ax[0])
dfit.qqplot(df['price'].values, n_top=n_top, ax=ax[1])


# %% ditfit
dfit = distfit(distr='full')
# Search for best theoretical fit on your empirical data
results = dfit.fit_transform(df['price'].values)
dfit.plot_summary(n_top=15)

n_top=1
fig, ax = plt.subplots(1,2, figsize=(25, 10))
dfit.plot(chart='PDF', n_top=n_top, ax=ax[0])
dfit.plot(chart='CDF', n_top=n_top, ax=ax[1])

fig, ax = plt.subplots(1,2, figsize=(25, 10))
dfit.plot_summary(ax=ax[0])
dfit.qqplot(df['price'].values, n_top=n_top, ax=ax[1])




# list(dfit.summary['name'][0:10].values)
from distfit import distfit
distr = ['johnsonsb', 'exponnorm', 'johnsonsu', 'invgauss', 'fatiguelife', 'exponweib', 'lognorm', 'invgamma', 'betaprime']
dfit = distfit(distr=distr)
results = dfit.fit_transform(df['price'].values)
dfit.plot_summary(n_top=10)


# %%
dfit = distfit(distr='johnsonsb')
dfit.fit_transform(df['price'].values)
dfit.predict(df['price'].values, alpha=0.05, multtest=None)
dfit.lineplot(df['price'], labels=df.index)

# %%


results = dfit.predict(df['price'].values, todf=True)
alpha = 0.05

outliers = results['df'].loc[results['P']<=alpha]
minth = outliers[outliers['y_pred']=='down']['y'].max()
maxth = outliers[outliers['y_pred']=='up']['y'].min()


fig, ax = plt.subplots(figsize=(15, 8))
x = df.index
y = df['price']
ax.plot(x, y, color='black')
ax.grid(False)

# ax.axhline(threshold, color='green', lw=2, alpha=0.7)
ax.fill_between(x, 0, 1, where=y >= maxth, color='green', alpha=0.5, transform=ax.get_xaxis_transform())
ax.fill_between(x, 0, 1, where=y <= minth, color='red', alpha=0.5, transform=ax.get_xaxis_transform())

# %% findpeaks
from findpeaks import findpeaks
# Initialize
fp = findpeaks(method='peakdetect')
# Detect peaks
results = fp.fit(df['price'].values)
# Plot
fp.plot()

# %%

# Import library
from pca import pca

# Initialize
model = pca()

# Load Titanic data set
df = model.import_example(data='titanic')

#      PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
# 0              1         0       3  ...   7.2500   NaN         S
# 1              2         1       1  ...  71.2833   C85         C
# 2              3         1       3  ...   7.9250   NaN         S
# 3              4         1       1  ...  53.1000  C123         S
# 4              5         0       3  ...   8.0500   NaN         S
# ..           ...       ...     ...  ...      ...   ...       ...
# 886          887         0       2  ...  13.0000   NaN         S
# 887          888         1       1  ...  30.0000   B42         S
# 888          889         0       3  ...  23.4500   NaN         S
# 889          890         1       1  ...  30.0000  C148         C
# 890          891         0       3  ...   7.7500   NaN         Q

# [891 rows x 12 columns]



# Intall onehot encoder
# pip install df2onehot
# Initialize
from df2onehot import df2onehot
df_clean = df.drop(labels=['PassengerId', 'Name', 'Cabin', 'Fare', 'Age', 'Ticket'], axis=1)

#      Survived_0.0  Survived_1.0  ...  Embarked_Q  Embarked_S
# 0            True         False  ...       False        True
# 1           False          True  ...       False       False
# 2           False          True  ...       False        True
# 3           False          True  ...       False        True
# 4            True         False  ...       False        True
# ..            ...           ...  ...         ...         ...
# 886          True         False  ...       False        True
# 887         False          True  ...       False        True
# 888          True         False  ...       False        True
# 889         False          True  ...       False       False
# 890          True         False  ...        True       False

# [891 rows x 25 columns]

df_hot = df2onehot(df_clean)['onehot']
# df_hot = df_hot.astype(int)
# df_hot = df_hot+np.random.random(df_hot.shape)/10
# df_hot = df_hot + np.random.normal(0, 0.01, size=df_hot.shape)


model = pca(normalize=True,
            detect_outliers=['ht2', 'spe'],
            alpha=0.05,
            n_std=3,
            multipletests='fdr_bh')

results = model.fit_transform(df_hot)

model.plot()
model.biplot(SPE=True,
             hotellingt2=True,
             jitter=0.1,
             n_feat=10,
             legend=True,
             label=False,
             y=df['Survived'],
             title='Survived',
             figsize=(20, 12),
             fontdict={'size':14},
             cmap=mpl.colors.ListedColormap(['red', 'blue']),
             gradient='#FFFFFF',
             )

results['outliers']
Iloc = np.logical_and(results['outliers']['y_bool'], results['outliers']['y_bool_spe'])
df.loc[Iloc]


#      PassengerId  Survived  Pclass  ...   Fare Cabin  Embarked
# 59            60         0       3  ...  46.90   NaN         S
# 71            72         0       3  ...  46.90   NaN         S
# 159          160         0       3  ...  69.55   NaN         S
# 180          181         0       3  ...  69.55   NaN         S
# 201          202         0       3  ...  69.55   NaN         S
# 324          325         0       3  ...  69.55   NaN         S
# 386          387         0       3  ...  46.90   NaN         S
# 480          481         0       3  ...  46.90   NaN         S
# 683          684         0       3  ...  46.90   NaN         S
# 792          793         0       3  ...  69.55   NaN         S
# 846          847         0       3  ...  69.55   NaN         S
# 863          864         0       3  ...  69.55   NaN         S

# [12 rows x 12 columns]

# from colourmap import colourmap
# colors = colourmap.fromlist(Iloc)[0]
# model.scatter(jitter=0.1, legend=True, label=False, y=df['Survived'], SPE=False, hotellingt2=False, title='Survived')
# model.scatter3d(jitter=0.1, legend=True, label=False, y=df['Survived'], SPE=False, hotellingt2=False, title='Survived')


# np.sum(model.results['outliers']['y_proba']<=0.05)



# for col in df.columns:
    # model.biplot(legend=True, label=False, y=df[col], SPE=True, hotellingt2=True, n_feat=10, title=col)


# from sklearn.manifold import TSNE
# xycoord = TSNE(n_components=2, init='random', perplexity=30).fit_transform(df_hot)
# from scatterd import scatterd
# scatterd(xycoord[:,0], xycoord[:,1], labels=df_hot['Survived_0.0'], legend=False, )

