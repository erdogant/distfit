""" This function is to save and load variables in/from pickle files."""

#----------------------------------
# Name        : picklefast.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
#----------------------------------

import pickle
import os


# %% Save pickle file
def save(filepath, var, verbose=3):
    """Saving pickle files for input variables.

    Parameters
    ----------
    filepath : str 
        Pathname to store pickle files.
    var : {list, object, dataframe, etc}
        Name of list or dict or anything else that needs to be stored.
    verbose : int, optional
        Show message. A higher number gives more informatie. The default is 3.

    Returns
    -------
    Status of succes : bool [True,False].


    Example
    -------
    import picklefast as pf
    filepath = './temp/tes1t.pkl'
    data = [1,2,3,4,5]
    status = picklefast.save(filepath, data)

    """
    # Make empty pickle file
    outfile = open(filepath,'wb')
    # Write and close
    pickle.dump(var,outfile)
    outfile.close()
    if os.path.isfile(filepath):
        if verbose>=3: print('[picklefast] Pickle file saved: [%s]' %filepath)
        out=True
    else:
        if verbose>=3: print('[picklefast] Pickle file could not be saved: [%s]' %filepath)
    return(out)


# %% Load pickle file
def load(filepath, verbose=3):
    """Loading pickle files for input variables.

    Parameters
    ----------
    filepath : str 
        Pathname to store pickle files.
    verbose : int, optional
        Show message. A higher number gives more informatie. The default is 3.

    Returns
    -------
    Status of succes : bool [True,False].


    Example
    -------
    import picklefast as pf
    filepath = './temp/tes1t.pkl'
    data = [1,2,3,4,5]
    status = picklefast.save(filepath, data)
    status = picklefast.load(filepath)

    """
    out=None
    if os.path.isfile(filepath):
        if verbose>=3: print('[picklefast] Pickle file loaded: [%s]' %filepath)
        pickle_off = open(filepath,"rb")
        out = pickle.load(pickle_off)
    else:
        if verbose>=3: print('[picklefast] Pickle file does not exists: [%s]' %filepath)    
    return(out)
