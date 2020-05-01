""" This function labels mixture of gaussians in an 1D array.

	A= dist2labx(data, <optional>)

 INPUT:
   data:           numpy-array

 OPTIONAL

   bins=           [Integer]: Bin size to make the estimation
                   None: (default)

   EMA:            [Integer]: Exponential Moving Average
                   None: (default)
                   20
                   50

   norm:           [Bool]: Normalize histogram data
                   False: No (default)
                   True: Yes

   alpha:          [Numeric]: Confidence interval boundaries
                   0.05: (default)

   mtyp:           [catagorical]: Histogram bins is created by numpy or seaborn functions
                   'numpy': (default)
                   'seaborn'

   fig_vertical:   [Bool]: Plot figure vertical
                   True: (default)
                   False: (horizontal)

   showfig:        Boolean [True,False]
                   False: No (default)
                   True: Yes

   verbose:        Boolean [True,False]


 OUTPUT
	output

 DESCRIPTION
   This function labels mixture of gaussians in an 1D array.

 EXAMPLE

   from sklearn.datasets.samples_generator import make_blobs
   [data, labels_true] = make_blobs(n_samples=10000, centers=3, n_features=1, cluster_std=0.3, random_state=0)

   A = dist2labx(data, mtype='numpy')
   A = dist2labx(data, mtype='seaborn')
   A = dist2labx(data)
   
   # Validation
   import SUPERVISED.confmatrix as confmatrix
   confmatrix.twoclass(labels_true==0, A['labx']==1)
   confmatrix.twoclass(labels_true==1, A['labx']==3)
   confmatrix.twoclass(labels_true==2, A['labx']==2)


   df = picklefast.load('D://stack/TOOLBOX_PY/DATA/STOCK/btc1h.pkl')['close']
   A=dist2labx(df.iloc[-20:])
   A=dist2labx(df.iloc[-20:], fig_vertical=False)
   A=dist2labx(df.iloc[-50:])
   A=dist2labx(df.iloc[-100:])
   A=dist2labx(df.iloc[-1000:], mtype='seaborn')
   A=dist2labx(df.iloc[-1000:], mtype='numpy')

   A=dist2labx(df.iloc[-200:], mtype='seaborn')
   A=dist2labx(df.iloc[-200:], mtype='numpy')
   A=dist2labx(df.iloc[-200:], alpha=0.001)
   A=dist2labx(df.iloc[-200:], bins=20)
   A=dist2labx(df.iloc[-500:], bins=50)


 SEE ALSO
   findpeaksvalleys
"""

#--------------------------------------------------------------------------
# Name        : dist2labx.py
# Author      : E.Taskesen
#--------------------------------------------------------------------------
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from findpeaks.boostdata1d import boostdata1d
from findpeaks.findpeaksvalleys import findpeaksvalleys


# %%
def dist2labx(data, bins=None, window=None, norm=False, alpha=0.05, mtype='seaborn', nboost=1000, showfig=True, fig_vertical=True, verbose=3):
	# DECLARATIONS
    out=dict()
    Param = {}
    Param['window'] = window
    Param['norm'] = norm
    Param['bins'] = bins
    Param['mtype'] = mtype
    Param['nboost'] = nboost
    Param['alpha'] = alpha
    Param['fig_vertical'] = fig_vertical
    Param['showfig'] = showfig
    Param['verbose'] = verbose

    # Set default bins
    if isinstance(bins, type(None)):
        Param['bins']='auto'

    # Boost data
    if not isinstance(Param['nboost'], type(None)):
        if Param['nboost']>len(data):
            data=boostdata1d(data, Param['nboost'], kind=2, showfig=False, verbose=Param['verbose'])

    # Moving average
    if not isinstance(Param['window'], type(None)):
        data = _EMA(data, window=Param['window'], showfig=showfig)

    # Check datatypes
    data = checkdatatypes(data)

    # Make histogram
    [histvals, binedges] = compute_histogram(data, Param)

    # Find the peaks and valleys of the expected mixture of gaussians
    peakdata = findpeaksvalleys(histvals, showfig=False, verbose=Param['verbose'])
    if isinstance(peakdata, type(None)):
        return(None)

    # Transform the detected peaks with the histograms back to the original data
    datalabx = compute_labx(data, binedges, peakdata)

    # Determine 95% CII per gaussian
    out = compute_CII(data, datalabx, Param)

    # Store data
    df = pd.DataFrame()
    df['histvals']   = histvals
    df['binedges']   = binedges[:-1]

    out['hist']      = df
    out['hist_labx'] = pd.DataFrame(peakdata['labxcol'])
    out['data']      = data.flatten()
    out['labx']      = datalabx
    out['bins']      = bins

    if Param['showfig']:
        if Param['fig_vertical']:
            makefig_vertical(out)
        else:
            makefig_horizontal(out)
            # makefig_horizontal1(out)

    return(out)

# %%
def _EMA(data, window, showfig=False):
    data = checktype(data)
    out = data.ewm(span=window).mean()
    if showfig: makefig(out)
    return out.values.flatten()

#%% Transform the detected peaks with the histograms back to the original data
def compute_labx(data, binedges, peakdata):
    datalabx=np.zeros(len(data))*np.nan
    clustnr = peakdata['labxcol']
    for i in range(0,clustnr.shape[1]):
        # Get the range in the histogram
        idxrange=np.where(np.isnan(clustnr[:,i])==False)[0]
        idx=np.where(np.logical_and(data>=binedges[idxrange].min(), data<=binedges[idxrange].max()))[0]
        datalabx[idx]=i+1
    return(datalabx)

#%% Determine 95% CII per gaussian
def compute_CII(data, datalabx, Param):
    out=dict()
    out['labx_info']=dict()

    uilabx = np.unique(datalabx)
    uilabx = uilabx[np.isnan(uilabx)==False]

    for labx in uilabx:
        I = datalabx==labx
        cii_high = pd.DataFrame(data[I]).quantile(q=1-Param['alpha']).values
        cii_low  = pd.DataFrame(data[I]).quantile(q=Param['alpha']).values
        mu       = pd.DataFrame(data[I]).mean().values
        median   = pd.DataFrame(data[I]).median().values
        out['labx_info'][int(labx)] = {'labx':np.where(I)[0], 'cii_low':cii_low[0], 'cii_high':cii_high[0], 'mu':mu[0], 'median':median[0]}
        if Param['verbose']>=3: print('[DIST2LABX] Labx %s, cii high:%s, cii low:%s' %(labx,cii_high,cii_low))

    return(out)

#%% 
def compute_histogram(data, Param):
    if Param['mtype']=='seaborn':
        [fig, ax]= plt.subplots()
        snsout=sns.distplot(data, bins=Param['bins'], norm_hist=Param['norm'], ax=ax).get_lines()[0].get_data()
        plt.close(fig)
        histvals=snsout[1]
        binedges=snsout[0]
        binedges=np.append(binedges,10**-6)
    elif Param['mtype']=='numpy':
        [histvals, binedges]=np.histogram(data, bins=Param['bins'], normed=Param['norm']) 
        binedges[-1] += 10**-6
    
    return([histvals, binedges])
    
#%% Check datatype
def checkdatatypes(data):
    if isinstance(data, type([])):
        data=np.array(data)
    if isinstance(data, type(pd.DataFrame())):
        data=data.values

    return(data)

#%% Make figure
def makefig_horizontal1(out):
    datalabx=out['labx']
    data=out['data']
    plt.figure()
    uilabx=np.unique(datalabx)
    uilabx = uilabx[np.isnan(uilabx)==False]
    for i in range(len(uilabx)):
        idx=uilabx[i]==datalabx
        plt.hist(data[idx],out['bins'])
        # plt.scatter(data[idx], np.zeros(data[idx].shape[0]),marker='x')
    plt.grid(True)

#%% Make figure
def makefig_horizontal(out):
    datalabx=out['labx']
    data=out['data']

    [fig, ax]=plt.subplots(2,1, figsize=(15,20))
    ax[0].plot(data)
    ax[0].grid(True)
    ax[0].set_ylim(np.min(data)*0.99, np.max(data)*1.01)
    ax[0].set_xlabel('ylabel')
    ax[0].set_ylabel('value')
    
    uilabx = np.unique(datalabx)
    uilabx = uilabx[np.isnan(uilabx)==False]
    colors = sns.color_palette('Set1',len(uilabx))

    sns.distplot(data, vertical=False, ax=ax[1], color=(0,0,0), bins=out['bins'])
    for i in range(len(uilabx)):
        idx=(uilabx[i]==datalabx)
        sns.distplot(data[idx], vertical=False, color=colors[i], ax=ax[1], bins=out['bins'])
        cii_low  = out['labx_info'][int(uilabx[i])]['cii_low']
        cii_high = out['labx_info'][int(uilabx[i])]['cii_high']
        median   = out['labx_info'][int(uilabx[i])]['median']
        ax[1].axvline(cii_low,  linestyle='--', c=colors[i], label='cii low')
        ax[1].axvline(cii_high, linestyle='--', c=colors[i], label='cii high')
        ax[1].axvline(median,   linestyle='-',  c=colors[i], label='mu', linewidth=2)
        ax[0].axhline(cii_low,  linestyle='--', c=colors[i], label='cii low')
        ax[0].axhline(cii_high, linestyle='--', c=colors[i], label='cii high')
        ax[0].axhline(median,   linestyle='-',  c=colors[i], label='mu', linewidth=2)
    
    ax[1].set_xlim(np.min(data)*0.99, np.max(data)*1.01)
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Frequency')
    ax[1].grid(True)

#%% Make figure
def makefig_vertical(out):
    datalabx=out['labx']
    data=out['data']

    [fig, ax]=plt.subplots(1,2, figsize=(25,8))
    ax[0].plot(data)
    ax[0].grid(True)
    ax[0].set_ylim(np.min(data)*0.99, np.max(data)*1.01)
    ax[0].set_ylabel('Value')
    ax[0].set_xlabel('Time')
    
    uilabx=np.unique(datalabx)
    uilabx = uilabx[np.isnan(uilabx)==False]
    colors=sns.color_palette('Set1',len(uilabx))

    sns.distplot(data, vertical=True, ax=ax[1], color=(0,0,0), bins=out['bins'])
    for i in range(len(uilabx)):
        idx=(uilabx[i]==datalabx)
        sns.distplot(data[idx], vertical=True, color=colors[i], ax=ax[1], bins=out['bins'])
        cii_low  = out['labx_info'][int(uilabx[i])]['cii_low']
        cii_high = out['labx_info'][int(uilabx[i])]['cii_high']
        median   = out['labx_info'][int(uilabx[i])]['median']
        ax[1].axhline(cii_low,  linestyle='--', c=colors[i], label='cii low')
        ax[1].axhline(cii_high, linestyle='--', c=colors[i], label='cii high')
        ax[1].axhline(median,   linestyle='-',  c=colors[i], label='mu', linewidth=2)
        ax[0].axhline(cii_low,  linestyle='--', c=colors[i], label='cii low')
        ax[0].axhline(cii_high, linestyle='--', c=colors[i], label='cii high')
        ax[0].axhline(median,   linestyle='-',  c=colors[i], label='mu', linewidth=2)
    
    ax[1].set_ylim(np.min(data)*0.99, np.max(data)*1.01)
#    ax[1].set_ylabel('Value')
    ax[1].set_xlabel('Frequency')
    ax[1].grid(True)
