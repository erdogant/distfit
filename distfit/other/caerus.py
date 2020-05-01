""" This function determines the local-minima with the corresponding local-maxima within the given time-frame

	A = caerus(df, <optional>)
	A = caerus_gridserch(df, <optional>)

 INPUT:
   data:           dataframe [nx1]
                   n=rows = time or other index
                   value
 OPTIONAL

   window:         Integer [1,...,len(data)] Window size that is used to determine whether there is an increase in percentage. start-location + window
                   50 (default) Smaller window size is able to pickup better local-minima
                   1000         Larger window size will more stress on global minma

   minperc:        Float [0,..,1] Minimum percentage to declare a starting position with window relevant. Note that nlargest is used to identify the top n largest percentages as stopping location.
                   3 (default)

   nlargest:       Integer [1,..,inf] Used to identify the top n percentages, and thus stop-regions (couples to start-region). The larger this number, the larger the stop-region untill it is limited by minperc.
                   10 (default)

   threshold:      Float [0,..,1] Required to optimize for the maximum depth of the local-minima. At the ith location, k windows (eg 50) are overlaid and the percentages are determined. The socre is determined by (percentage(i-start,k-stop)) >= minperc (eg 3), and normalized for the maximum number of windows used at position i. In best case scenarion, all window result in percentage>minperc and will hve score 50/50=1. 
                   0.25 (default)

   verbose:        Boolean [0,1]
                   0: No (default)
                   1: Yes

 OUTPUT
	output

 DESCRIPTION
    In Greek mythology, Caerus (same as kairos) was the personification of opportunity, luck and favorable moments. 
    He was shown with only one lock of hair. His Roman equivalent was Occasio or Tempus. Caerus was the youngest child of Zeus.

    This function determines the local-minima with the corresponding local-maxima within the given time-frame.
    The method is as following; in a forward rolling window, thousands of windows are 
    iteratively created and for each window a percentage score is computed from the start-to-stop position.
    For resulting matrix [window x length dataframe], only the high scoring percentages, e.g. those above a certain value (minperc) are used.
    The cleaned matrix is then aggregated by sum per time-point followed by a cut using the threshold.
    The resulting regions are subsequently detected, and represent the starting-locations of the trade.
    The stop-locations are determined based on the distance and percentage of te start-locations.
    As an example, if you want to have best regions, use threshold=1, minperc=high and nlargest=1 (small)

 EXAMPLE
    %reset -f
    from STATS.caerus import caerus, caerus_gridserch
    import numpy as np

    df = picklefast.load('../DATA/STOCK/btcyears.pkl')['Close']
    df = picklefast.load('../DATA/STOCK/btc1h.pkl')['close']
    out=caerus(df, window=50, minperc=3, threshold=0.25, nlargest=10)

    # Best parameters
    [out_balance, out_trades]=caerus_gridserch(df)

    # Shuffle
    df = picklefast.load('D://stack/TOOLBOX_PY/DATA/STOCK/btc1h.pkl')['close']
    np.random.shuffle(df)
    outNull=caerus(df, window=50, minperc=3, nlargest=10, threshold=0.25)
    plt.figure();plt.hist(outNull['agg'], bins=50)
    Praw=hypotesting(out['agg'], outNull['agg'], showfig=0, bound='up')['Praw']
    model=distfit(outNull['agg'], showfig=1, alpha=0.05)[0]
    

 SEE ALSO

"""

#--------------------------------------------------------------------------
# Name        : caerus.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : June. 2019
#--------------------------------------------------------------------------

#%% Libraries
import pandas as pd
import numpy as np
import STATS.percentage as percentage
import matplotlib.pyplot as plt
from GENERAL.ones2idx import ones2region, idx2region, region2idx
from STOCKS.risk_performance_metrics import risk_performance_metrics

#%% Detection of optimal localities for investments
def caerus(df, window=50, minperc=3, nlargest=10, threshold=0.25, extb=0, extf=10, showplot=True, verbose=3):
    Param = dict()
    Param['verbose']   = verbose
    Param['window']    = window
    Param['minperc']   = minperc
    Param['threshold'] = threshold
    Param['nlargest']  = nlargest
    Param['extb']      = extb
    Param['extf']      = extf

    # Convert to dataframe
    if 'numpy' in str(type(df)) or 'list' in str(type(df)):
        df = pd.DataFrame(df, columns=['data'])

    # Some checks
    # assert isinstance(df, pd.DataFrame), 'Input data must be of type <pd.DataFrame()>'
    # assert df.shape[1]==1, 'Input data can only have 1 column with data'
    
    # reset index
    df.reset_index(drop=True, inplace=True)
    # Run over all windows
    out = compute_region_scores(df, window=Param['window'], verbose=Param['verbose'])
    # Keep only percentages above minimum
    out = out[out>Param['minperc']]
    # Find local minima-start-locations
    [loc_start, outagg] = regions_detect_start(out, Param['minperc'], Param['threshold'], extb=Param['extb'], extf=Param['extf'])
    # Find regions that are local optima for the corrersponding local-minima
    loc_stop = regions_detect_stop(out, loc_start, Param['nlargest'], extb=Param['extb'], extf=Param['extf'], verbose=Param['verbose'])
    # Find regions that are local optima for the corrersponding local-minima
    [loc_start_best, loc_stop_best] = get_locs_best(df, loc_start, loc_stop)
    # Make figure
    if showplot:
        makefig(df, loc_start, loc_stop, loc_start_best, loc_stop_best, out, threshold=Param['threshold'])

    out=dict()
    out['loc_start']=loc_start
    out['loc_stop']=loc_stop
    out['loc_start_best']=loc_start_best
    out['loc_stop_best']=loc_stop_best
    out['agg']=outagg

    return(out)

#%% Merge regions
def get_locs_best(df, loc_start, loc_stop):
    loc_start_best=np.zeros(len(loc_start)).astype(int)
    loc_stop_best=np.zeros(len(loc_start)).astype(int)
    for i in range(0,len(loc_start)):
        loc_start_best[i]=df.iloc[loc_start[i][0]:loc_start[i][1]+1].argmin()

        tmpvalue=pd.DataFrame()
        for k in range(0,len(loc_stop[i])):
            idx_start=np.minimum(loc_stop[i][k][0], df.shape[0]-1)
            idx_stop=np.minimum(loc_stop[i][k][1]+1, df.shape[0])
            tmpvalue = pd.concat((tmpvalue, df.iloc[idx_start:idx_stop]))

        loc_stop_best[i]=tmpvalue.idxmax()[0]
    return(loc_start_best, loc_stop_best)
    
#%% Merge regions
def regions_merge(data, extb=5, extf=5):
    out=None
    if not isinstance(data,type(None)):
        data    = np.array(data)
        idx     = np.argsort(data[:,0])
        data    = data[idx,:]
        loc_bin = np.zeros(np.max(data)+1)

        # Add ones to array
        if data.shape[0]==1:
            loc_bin[np.arange(data[0][0], data[0][1]+1)]=1
        else:
            for i in range(0,len(data)):
                if i<len(data)-1:
                    # Check whether in each others range
                    if data[i][1]+extf>=data[i+1][0]-extb:
                        XtraOnes=np.arange(data[i][1], data[i+1][0])
                        # Add ones to array
                        loc_bin[XtraOnes]=1
                
                # Add ones to array
                loc_bin[np.arange(data[i][0], np.minimum(data[i][1]+1, len(loc_bin)))]=1
            
        # Find the merged indexes
        out = ones2region(loc_bin)
    return(out)

#%% Compute scores using a forward rolling window
def compute_region_scores(df, window=1000, verbose=0):
    # Compute percentage score for each 
    # 1. Window
    # 2. Position
    
    # Start with empty dataframe
    out=pd.DataFrame()

    # Reverse dataframe to create forward-rolling window
    df=df[::-1]
    for i in tqdm(range(2,window), disable=(True if verbose==0 else False)):
        dfperc = df.rolling(i).apply(compute_percentage)[::-1] #.values.flatten()
        out[i]=dfperc
    
    out[np.isinf(out)]=np.nan
    # out.fillna(value=0, inplace=True)

    return(out)

#%% Aggregation of scores over the windows and intervals
def agg_scores(out, threshold=0):
    outagg=np.nansum(out>0, axis=1)
    # Normalize for the window size that is used. Towards the end smaller windows are only avaialbe which is otherwise unfair for the threshold usage.
    windowCorrectionFactor=np.ones_like(outagg)*out.shape[1]
    tmpvalue=np.arange(1,out.shape[1])[::-1]
    windowCorrectionFactor[-len(tmpvalue):]=tmpvalue

    outagg = outagg/windowCorrectionFactor
    I=outagg>threshold
    return(outagg, I)
    
#%% Detect starting positions for regions
def regions_detect_start(out, minperc, threshold, extb=5, extf=5):
    # Find start-locations
    [outagg, I] = agg_scores(out, threshold)
    locs_start=ones2region(I)
    
    if len(locs_start)==0:
        locs_start=None

    # Merge regions if only seperated with few intervals
    locs_start = regions_merge(locs_start, extb=extb, extf=extf)

    return(locs_start, outagg)

#%% Detect stop locations based on the starting positions
def regions_detect_stop(out, locs_start, nlargest, extb=5, extf=5, verbose=0):
    # Find stop-locations
    locs_stop=None
    if not isinstance(locs_start,type(None)):

        locs_stop=[]
#        out[np.isinf(out)]=np.nan
        
        for i in range(0,len(locs_start)):
            if verbose>=4: print('[CAERUS] Region %s' %(i))
            # Take window sizes with maximum percentages
            # getloc=out.iloc[locs_start[i][0]:locs_start[i][1]+1,:].idxmax(axis=1)

            # Get window size and add to starting indexes
            startlocs= np.arange(locs_start[i][0], locs_start[i][1]+1)
            
            getloc=[]
            getpos=out.iloc[locs_start[i][0]:locs_start[i][1]+1,:]
            
            # Run over all positions to find the top-n maximum ones
            for k in range(0,getpos.shape[0]):
                tmplocs = getpos.iloc[k,:].nlargest(nlargest).index.values
                tmplocs = tmplocs+startlocs[k]
                getloc=np.append(np.unique(getloc), tmplocs)
            
            getloc = np.sort(np.unique(getloc)).astype(int)

            # Merge if required
            getloc=idx2region(getloc)
            getloc=regions_merge(getloc, extb=extb, extf=extf)

            # Compute mean percentages per region and sort accordingly
            loc_mean_percentage=[]
            for p in range(0,len(getloc)):
                loc_mean_percentage.append(np.nanmean(out.iloc[getloc[p][0]:getloc[p][1]+1,:]))
            loc_mean_percentage=np.array(loc_mean_percentage)
            idx=np.argsort(loc_mean_percentage)[::-1]
            getloc=np.array(getloc)[idx]
            
            locs_stop.append(getloc.tolist())

    return(locs_stop)

#%% Make final figure
def makefig(df, loc_start, loc_stop, loc_start_best, loc_stop_best, out, threshold=0.3):
    [fig,(ax1,ax2,ax3)]=plt.subplots(3,1)
    # Make heatmap
    ax1.matshow(out.T)
    ax1.set_aspect('auto')
    # ax1.gca().set_aspect('auto')
    ax1.grid(True)
    
    # make aggregated figure
    # Normalized successes across the n windows for percentages above minperc.
    # 1 depicts that for location i, all of the 1000 windows of different length was succesfull in computing a percentage above minperc
    [outagg, I] = agg_scores(out, threshold)
    ax2.plot(outagg)
    ax2.grid(True)

    # Plot local minima-maxima
    ax3.plot(df.iloc[loc_start_best],'og', linewidth=1)
    ax3.plot(df.iloc[loc_stop_best],'or', linewidth=1)

    # Plot region-minima-maxima
    ax3.plot(df,'k', linewidth=1)
    for i in range(0,len(loc_start)):
        ax3.plot(df.iloc[np.arange(loc_start[i][0],loc_start[i][1])],'g', linewidth=2)
        # ax3.plot(df.iloc[np.arange(loc_stop[i][0],loc_stop[i][1])],'r', linewidth=2)
        # ax3.plot(df.iloc[loc_stop[i]], 'or', linewidth=2)
        for k in range(0,len(loc_stop[i])):
            ax3.plot(df.iloc[np.arange(loc_stop[i][k][0],loc_stop[i][k][1])],'r', linewidth=2)

    ax3.grid(True)
    plt.show()
    return

#%% Compute percentage
def compute_percentage(r):
    perc=percentage.getdiff(r[0],r[-1])
    return(perc) 

#%% Perform gridsearch to determine best parameters
def caerus_gridserch(df, window=None, perc=None, threshold=0.25, showplot=True, verbose=3):
    if verbose>=3: print('[CAERUS] Gridsearch..')

    if isinstance(window, type(None)):
        windows=np.arange(50,550,50)
    if isinstance(perc, type(None)):
        perc=np.arange(0,1,0.1)
    if showplot:
        [fig,(ax1,ax2)]=plt.subplots(2,1)

    out_balance = np.zeros((len(perc),len(windows)))
    out_trades  = np.zeros((len(perc),len(windows)))

    for k in range(0,len(windows)):
        for i in tqdm(range(0,len(perc), disable=(True if verbose==0 else False))):
            # Compute start-stop locations
            getregions=caerus(df, window=windows[i], minperc=perc[k], threshold=threshold, nlargest=1, showplot=False, verbose=0)
            # Store
            perf=pd.DataFrame()
            perf['portfolio_value'] = df.values.copy()
            perf['asset']           = df.values.copy()
            perf['invested']        = 0
            perf['invested'].iloc[region2idx(np.vstack((getregions['loc_start_best'], getregions['loc_stop_best'])).T)]=1
            performanceMetrics = risk_performance_metrics(perf)
            # Compute score
            out_balance[i,k] =performanceMetrics['winning_balance']
            out_trades[i,k]=performanceMetrics['winning_trades']

        if showplot:
            #label = list(map(( lambda x: 'window_' + x), windows.astype(str)))
            ax1.plot(perc,out_balance[:,k], label='window_'+str(windows[k]))
            ax2.plot(perc,out_trades[:,k], label='window_'+str(windows[k]))

        # showprogress(k,len(windows))

    if showplot:
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel('Percentage')
        ax1.set_ylabel('winning_balance')
        ax2.grid(True)
        ax2.set_xlabel('Percentage')
        ax2.set_ylabel('Nr Trades')
        plt.show()
    
    out_balance  = pd.DataFrame(index=perc, data=out_balance, columns=windows)
    out_trades   = pd.DataFrame(index=perc, data=out_trades, columns=windows)
    return(out_balance, out_trades)

#%% End