""" This function checks 89 different distributions and computes which one fits best to your emperical distribution based on Mean Square error (MSE) estimates.

    import distfit as distfit

    out    = distfit.parametric(data, <optional>)
    fig,ax = distfit.plot(out, <optional>)


 INPUT:
   data:           Numpy array
                   

 OPTIONAL

   bins=           [Integer]: Bin size to make the estimation
                   [50]: (default)

   alpha=          Double : [0..1] Significance alpha
                   [None]: Default
   
   bound=          String: Set whether you want returned a P-value for the lower/upper bounds or both
                   'both': Both (default)
                   'up':   Upperbounds
                   'down': Lowerbounds

   distribution=   String: Set the distribution to use
                   'auto_small': A smaller set of distributions: [norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm] (default) 
                   'auto_full' : The full set of distributions
                   'norm'      : normal distribution
                   't'         : Students T distribution
                   etc

   title=          String Title of the figure
                   '' (default)

   showfig=        [Boolean] [0,1]: Show figure
                   [0]: No
                   [1]: Yes (Default)

   verbose=   [Boolean] [0,1]
                   [0]: No (default)
                   [1]: Yes
                   [2]: Yes (More information)

 OUTPUT
	dictionary with best ditribution


 #EXAMPLE
   import numpy as np
   import distfit as distfit


   #=============== Find best fit distribution ==============================
   data=np.random.beta(5, 8, 10000)
   out = distfit.parametric(data)
   distfit.plot(out)

   data=np.random.normal(5, 8, 10000)
   out = distfit.parametric(data, bins=50, distribution='auto_full')
   distfit.plot(out)

   data=np.random.normal(5, 8, [100,100])
   out = distfit.parametric(data, distribution='auto_small')
   distfit.plot(out)

   data=np.random.normal(5, 8, [100,100])
   out = distfit.parametric(data, distribution='norm')
   distfit.plot(out)

   
 SEE ALSO: hypotesting
"""

#--------------------------------------------------------------------------
# Name        : distfit.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Sep. 2017
#--------------------------------------------------------------------------


#%% Libraries
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from distfit.helpers.hist import hist
import seaborn as sns
#from scipy.optimize import curve_fit

#%%
def format_data(data):
    # Convert pandas to numpy
    if str(data.dtype)=='O': data=data.astype(float)
    if 'pandas' in str(type(data)): data = data.values
    # Make sure its a vector
    data = data.ravel()
    return(data)

def get_distributions(distribution):
    DISTRIBUTIONS=[]
    # Distributions to check
    if distribution=='auto_full':
        DISTRIBUTIONS = [st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.laplace,st.levy,st.levy_l,st.levy_stable,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,
            st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
    elif distribution=='auto_small':
        DISTRIBUTIONS = [st.norm, st.expon, st.pareto, st.dweibull, st.t, st.genextreme, st.gamma, st.lognorm, st.beta, st.uniform]
    else:
        # Connect object with variable to be used as a function again.
        DISTRIBUTIONS  = [getattr(st, distribution)]

    return(DISTRIBUTIONS)

#%% Get histogram of original data
def get_hist_params(data, bins):
    [y_obs, X] = np.histogram(data, bins=bins, density=True)
    X = (X + np.roll(X, -1))[:-1] / 2.0
    #plt.plot(X,y_obs)
    return(y_obs, X)

#%% Compute score for each distribution
def compute_score_distribution(data, y_obs, X, DISTRIBUTIONS, verbose=3):
    out      = []
    out_dist = {}
    out_dist['distribution'] = st.norm
    out_dist['params'] = (0.0, 1.0)
    best_sse = np.inf
    out = pd.DataFrame(index=range(0,len(DISTRIBUTIONS)), columns=['Distribution','SSE','LLE','loc','scale','arg'])
    
    # Estimate distribution parameters
    # i=0
    for i,dist in enumerate(DISTRIBUTIONS):
        logLik=0

        # Try to fit the dist. However this can result in an error so therefore you need to try-catch
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = dist.fit(data)

                # Separate parts of parameters
                arg   = params[:-2]
                loc   = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = dist.pdf(X, loc=loc, scale=scale, *arg)
                # Compute SSE
                sse = np.sum(np.power(y_obs - pdf, 2.0))

                # Compute Maximum likelhood
#                x = np.linspace(0, 100, num=100)
#                y_obs = 5 + 2.4*x + np.random.normal(0, 4, len(data))
#                b0 = params[0]
#                b1 = params[1]
#                nu = params[2]
#                yPred = b0 + b1*x
#                logLik = -np.log( np.prod(pdf(x, mu=yPred, nu=nu)))
                
#                logLik=0
                # Calculate the negative log-likelihood as the negative sum of the log of a normal
                # PDF where the observed values are normally distributed around the mean (yPred)
                # with a standard deviation of sd
                # Calculate the predicted values from the initial parameter guesses
#                yPred = params[0] + params[1]*X
                try:
                    logLik = -np.sum( dist.logpdf(y_obs, loc=loc, scale=scale) )
                except Exception:
                    logLik = float('NaN')
                    pass
#                if len(params)>2:
#                    logLik = -np.sum( dist.logpdf(y_obs, arg=arg, loc=loc, scale=scale) )
#                else:
#                    logLik = -np.sum( dist.logpdf(y_obs, loc=loc, scale=scale) )
                
#                # Store results
                out.values[i,0] = dist.name
                out.values[i,1] = sse
                out.values[i,2] = logLik
                out.values[i,3] = loc
                out.values[i,4] = scale
                out.values[i,5] = arg
                
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_sse                 = sse
                    out_dist['name']         = dist.name
                    out_dist['distribution'] = dist
                    out_dist['params']       = params
                    out_dist['sse']          = sse
                    out_dist['loc']          = loc
                    out_dist['scale']        = scale
                    out_dist['arg']          = arg

            if verbose>=3:
                print("[DISTFIT] Checking for [%s] [SSE:%f] [logLik:%f]" %(dist.name,sse,logLik))
            
        except Exception:
            pass    

    # Sort the output
    out = out.sort_values('SSE')
    out.reset_index(drop=True, inplace=True)
    # Return
    return(out, out_dist)

#%% Determine confidence intervals on the best fitting distribution
def compute_cii(out_dist, alpha=None, bound='both'):
    # Separate parts of parameters
    arg   = out_dist['params'][:-2]
    loc   = out_dist['params'][-2]
    scale = out_dist['params'][-1]

    # Determine %CII
    dist   = getattr(st, out_dist['name'])
    CIIup, CIIdown  = None, None
    if alpha!=None:
        if bound=='up' or bound=='both' or bound=='right':
            CIIdown = dist.ppf(1-alpha, *arg, loc=loc, scale=scale) if arg else dist.ppf(1-alpha, loc=loc, scale=scale)
        if bound=='down' or bound=='both' or bound=='left':
            CIIup = dist.ppf(alpha, *arg, loc=loc, scale=scale) if arg else dist.ppf(alpha, loc=loc, scale=scale)
    
    # Store
#    out_dist['CII_min_'+str(alpha)]=CIIup
#    out_dist['CII_max_'+str(alpha)]=CIIdown
    out_dist['CII_min_alpha']=CIIup
    out_dist['CII_max_alpha']=CIIdown
    
    return(out_dist)

#%% Main
def parametric(data, bins=50, distribution='auto_small', alpha=None, bound='both', verbose=3): 
	# DECLARATIONS
    Param    = {}
    Param['verbose']      = verbose
    Param['bins']         = bins
    Param['distribution'] = distribution
    Param['alpha']        = alpha
    Param['bound']        = bound

    assert len(data)>0, print('[DISTFIT] Data vector is empty')

    # Format the data
    data = format_data(data)

    # Get list of distributions to check
    DISTRIBUTIONS = get_distributions(Param['distribution'])

    # Get histogram of original data
    [y_obs, X] = get_hist_params(data, Param['bins'])
    
    # Compute best distribution fit on the emperical data
    out_summary, model = compute_score_distribution(data, y_obs, X, DISTRIBUTIONS, verbose=Param['verbose'])
    
    # Determine confidence intervals on the best fitting distribution
    model = compute_cii(model, alpha=Param['alpha'], bound=Param['bound'])
    
    # Return
    out=dict()
    out['Param']=Param
    out['data']=data
    out['model']=model
    out['summary']=out_summary
    return(out)

#%% Plot
def plot(out, title='', width=8,  height=8, xlim=[], ylim=[], showfig=2, alpha=None, verbose=3):
    out_dist = out['summary']
    out_dist = out['model']
    data = out['data']
    
    Param = out['Param']
    Param['showfig'] = showfig
    Param['title'] = title
    Param['width'] = width
    Param['height'] = height
    Param['xlim'] = xlim
    Param['ylim'] = ylim
    
    if not alpha==None:
        Param['alpha'] = alpha

    # Make figure
    best_dist = out_dist['distribution']
    best_fit_name = out_dist['name']
    best_fit_param = out_dist['params']
    arg   = out_dist['params'][:-2]
    loc   = out_dist['params'][-2]
    scale = out_dist['params'][-1]
    dist   = getattr(st, out_dist['name'])
    size  = len(data)

    out_dist['ax']=None
    if Param['showfig']==1:
        # Plot line
        getmin = dist.ppf(0.0000001, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.0000001, loc=loc, scale=scale)
        getmax = dist.ppf(0.9999999, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.9999999, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x   = np.linspace(getmin, getmax, size)
        y   = dist.pdf(x, loc=loc, scale=scale, *arg)

        # plt.figure(figsize=(6,4))
        [fig, ax]=hist(data,bins=Param['bins'],xlabel='Values',ylabel='Frequency', grid=True, normed=1, verbose=Param['verbose']>=1, width=Param['width'],height=Param['height'])
        plt.plot(x, y, 'b-', linewidth=2)
        legendname=[best_fit_name,'Emperical distribution']
    
        # Plot vertical line To stress the cut-off point
        if not Param['alpha']==None:
            if Param['bound']=='down' or Param['bound']=='both':
                ax.axvline(x=out_dist['CII_min_alpha'], ymin=0, ymax=1, linewidth=2, color='r', linestyle='dashed')
                legendname=[best_fit_name,'CII low '+'('+str(Param['alpha'])+')', 'Emperical distribution']

            if Param['bound']=='up' or Param['bound']=='both':
                ax.axvline(x=out_dist['CII_max_alpha'], ymin=0, ymax=1, linewidth=2, color='r', linestyle='dashed')
                legendname=[best_fit_name,'CII high '+'('+str(Param['alpha'])+')', 'Emperical distribution']

            if Param['bound']=='both':
                legendname=[best_fit_name,'CII low '+'('+str(Param['alpha'])+')','CII high '+'('+str(Param['alpha'])+')','Emperical distribution']
        
        plt.legend(legendname)
        # Make text for plot
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str   = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_param)])
        dist_str    = '{}({})'.format(best_fit_name, param_str)
        ax.set_title(Param['title']+'\nBest fit distribution\n' + dist_str)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')    

        #% Limit axis
        if Param['xlim']!=[]:
            plt.xlim(xmin=Param['xlim'][0], xmax=Param['xlim'][1])
        if Param['ylim']!=[]:
            plt.ylim(ymin=Param['ylim'][0], ymax=Param['ylim'][1])

        #Store axis information
        out_dist['ax']=ax
    
    # Make figure
    if Param['showfig']==2:
        [fig,ax]=plt.subplots(figsize=(Param['width'],Param['height']))
        sns.distplot(data, bins=Param['bins'], hist=True, kde=True, rug=False, color = 'darkblue', kde_kws={'linewidth': 3}, rug_kws={'color': 'black'}, label='Bins')
        # Make text for plot
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str   = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_param)])
        dist_str    = '{}({})'.format(best_fit_name, param_str)
        plt.title(Param['title']+'\nBest fit distribution\n' + dist_str)
        plt.xlabel('Values')
        plt.ylabel('Frequency') 
        plt.grid()

        #% Limit axis
        if Param['xlim']!=[]:
            plt.xlim(xmin=Param['xlim'][0], xmax=Param['xlim'][1])
        if Param['ylim']!=[]:
            plt.ylim(ymin=Param['ylim'][0], ymax=Param['ylim'][1])
        
    if Param['verbose']>=1:
        print("[DISTRIBUTION FIT] Estimated distribution: %s [loc:%f, scale:%f]" %(out_dist['name'],out_dist['params'][-2],out_dist['params'][-1]))
        
    return (fig, ax)
    #return (out_dist, best_dist.name, best_params, out)
