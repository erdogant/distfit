""" This function saves figures in PNG format.

	A=savefig(data, <optional>)

 INPUT:
   data:           fig object

 OPTIONAL

   showprogress   : Boolean [0,1]
                   [0]: No (default)
                   [1]: Yes

 OUTPUT
	BOOLEAN
    [0]: If not succesful 
    [1]: If succesful 

 DESCRIPTION
   his function saves figures in PNG format.

 EXAMPLE
   %reset -f
   print(os.getcwd())
   from donutchart import donutchart

   A = donutchart([15, 30, 45, 10],['aap','boom','mies','banaan'])
   B = savefig(A,"c://temp//magweg//fig.png",showprogress=1)

 SEE ALSO
   
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : savefig.py
# Version     : 1.0
# Author      : E.Taskesen
# Date        : Sep. 2017
#--------------------------------------------------------------------------
# Libraries
from os import mkdir
from os import path

#%% Main
def savefig(fig, filepath, dpi=100, transp=False, showprogress=0):
    out=0 # Returns 1 if succesful
    # Make dictionary to store Parameters
    Param = {}
    Param['showprogress'] = showprogress
    Param['filepath']     = filepath
    Param['dpi']          = dpi
    Param['transp']       = transp

    # Write figure to path
    if Param['filepath']!="":
        # Check dir
        [getpath, getfilename] = path.split(Param['filepath'])
        if path.exists(getpath)==False:
            mkdir(getpath)
        #end
        
        #save file
        #print(fig.canvas.get_supported_filetypes())
        fig.savefig(Param['filepath'], dpi=Param['dpi'], transparent=Param['transp'], bbox_inches='tight')
        out=1
        
    return(out)
