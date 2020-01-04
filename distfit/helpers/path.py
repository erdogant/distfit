#out=dirwalk('D:/magweg/', extkeep=['.jpg'])

#%% Libraries
import os
import numpy as np
import re
from pathlib import Path

#%% Walk through directory
def dirwalk(directory, ext=None, exclude=None, recursive=True):
    # Get all files in directory
    if recursive:
        out = do_recursive(directory)
        out = np.array(list(out))
    else:
        out = os.path.join(directory, '')+np.array(os.listdir(directory)).astype('O')
    
    # Remove exclude path
    if not isinstance(exclude, type(None)):
        if (exclude[-1]=='/'): exclude=exclude[:-1]
        I=np.array(list(map(lambda x: re.findall(exclude,x)==[],out)))
        out=out[I]
    
    # Filter
    if not isinstance(ext, type(None)):
        # Split
        file_ext=np.array(list(map(split, out)))[:,-1]
        # Lower
        file_ext=list(map(str.lower, file_ext))
        # Keep only extentions
        out=out[np.isin(file_ext, ext)]
    
    return(out)

#%% Run recursivly through the directory
def do_recursive(directory):
    "walk a directory tree, using a generator"
    for name in os.listdir(directory):
        fullpath = os.path.join(directory, name)
        if os.path.isdir(fullpath) and not os.path.islink(fullpath):
            for name in do_recursive(fullpath):  # recurse into subdir
                yield name
        else:
            yield fullpath

#%% Split filepath into dir, filename and extension
def split(filepath, rem_spaces=False):
    [dirpath, filename]=os.path.split(filepath)
    [filename,ext]=os.path.splitext(filename)
    if rem_spaces:
        filename=filename.replace(' ','_')
    return(dirpath, filename, ext)

#%% Split filepath into dir, filename and extension
def split1d(filepath, rem_spaces=False):
    return(np.array(list(map(split, filepath))))

#%% Extract filename with extension from filepath
def filepaths2filenames1d(filepaths, with_extension=True):
    # Extract only filenames
    filenames=np.array(list(map(split, filepaths)), dtype='O')[:,1]
    exts=np.array(list(map(split, filepaths)))[:,2]
    if with_extension:
        filenames=filenames+exts
    return(filenames)

#%% Extract subdirectory name where the file is located
def lastdir(filepath):
    return(split(split(filepath)[0])[1])
    
#%% Extract subdirectory name where the file is located
def lastdir1d(filepaths):
    return(split1d(split1d(filepaths)[:,0])[:,1])

#%% From savepath to full and correct path
def correct(savepath, filename='fig', ext='.png'):
    '''
    savepath can be a string that looks like below.
    savepath='./tmp'
    savepath='./tmp/fig'
    savepath='./tmp/fig.png'
    savepath='fig.png'
    '''
    out=None
    if not isinstance(savepath, type(None)):
        # Set savepath and filename
        [getdir,getfile,getext]=split(savepath)
        # Make savepath
        if len(getdir[1:])==0: getdir=''
        if len(getfile)==0: getfile=filename
        if len(getext)==0: getext=ext
        # Create dir
        if len(getdir)>0:
            path=Path(getdir)
            path.mkdir(parents=True, exist_ok=True)
        # Make final path
        out=os.path.join(getdir,getfile+getext)
    return(out)