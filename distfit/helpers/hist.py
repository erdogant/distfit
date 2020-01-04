""" This function makes histogram

	fig,ax = hist(data, <optional>)

 INPUT:
   data:           datamatrix
                   rows    = features
                   colums  = samples
 OPTIONAL

   bins:           Integer:  Number of bins
                   [50]: (default)

   xlabel:         String: strings for the xlabel
                   'Xlabel'

   ylabel:         String: strings for the ylabel
                   'Ylabel'

   title:         String: strings for the title
                   ''

   height:         Integer:  Height of figure
                   [10]: (default)

   width:          Integer:  Width of figure
                   [10]: (default)

   grid:           Boolean [0,1]: Grid in figure
                   'True' (default)
                   'False'

   facecolor:      String: strings for the facecolor
                   'k' (default)

   savepath:       String: pathname of the file
                   'c:/temp/heatmap.png'

   dpi:            Integer: Resolution of the figure
                   [100] (default)

   verbose    Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes

 OUTPUT
	[fig, plt, ax]


 DESCRIPTION
   Makes histogram in matplotlib
   https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.hist.html
   https://matplotlib.org/api/matplotlib_configuration_api.html

 EXAMPLE
   import numpy as np
   import hist as bdr

   savepath = "fig.png"
   data=np.random.normal(0, 0.5, 1000)
   [plt,fig, ax] = hist(data,savepath=savepath)

"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : hist.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Sep. 2017
#--------------------------------------------------------------------------

#%% Libraries
import matplotlib.pyplot as plt
from distfit.helpers.savefig import savefig
import distfit.helpers.path as path
import warnings
warnings.filterwarnings("ignore")

#%% Main
def hist(data, bins=50, xlabel="x-axis", ylabel="Frequency", title="", width=10, height=10, savepath="", facecolor='k', grid=True, dpi=100, normed=False, verbose=0):
    fig =[];
    Param = {}
    Param['verbose'] = verbose
    Param['bins']         = bins
    Param['title']        = title
    Param['xlabel']       = xlabel
    Param['ylabel']       = ylabel
    Param['width']        = width
    Param['height']       = height
    Param['facecolor']    = facecolor
    Param['grid']         = grid
    Param['dpi']          = dpi
    Param['normed']       = normed
    Param['savepath']     = savepath

    # Make Figure
    SMALL_SIZE  = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14
    
    plt.rc('font',  size       = SMALL_SIZE)   # controls default text sizes
    plt.rc('axes',  titlesize  = SMALL_SIZE)   # fontsize of the axes title
    plt.rc('xtick', labelsize  = SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('ytick', labelsize  = SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize  = SMALL_SIZE)   # legend fontsize
    plt.rc('axes',  labelsize  = MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

    axis_font = {'fontname':'Arial'}

    # Make figure with properties
    [fig,ax]=plt.subplots()

    # specify dimensions
    fig.set_size_inches(Param['width'], Param['height'])

    # Histogram plot
    [n, bins, patches] = plt.hist(data, Param['bins'], normed=Param['normed'], facecolor=Param['facecolor'], alpha=0.5)
    
    # Fig settings
    plt.xlabel(Param['xlabel'], **axis_font)
    plt.ylabel(Param['ylabel'], **axis_font)
    plt.title(Param['title'])
    plt.grid(Param['grid'])
    plt.show()

    # Write figure to path
    if not Param['savepath']=='':
        savepath=path.correct(Param['savepath'])
        savefig(fig, savepath, dpi=Param['dpi'])

    return(fig, ax)
