# ----------------------------------------------------
# Name        : smoothline.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# ----------------------------------------------------

import numpy as np
from scipy.interpolate import make_interp_spline

def smoothline(xs, ys=None, interpol=3, window=1, verbose=3):
    """Smoothing 1D vector.
    
    Description
    -----------
    Smoothing a 1d vector can be challanging if the number of data is low sampled.
    This smoothing function therefore contains two steps. First interpolation of the 
    input line followed by a convolution.

    Parameters
    ----------
    xs : array-like
        Data points for the x-axis.
    ys : array-like
        Data points for the y-axis.
    interpol : int, (default : 3)
        The interpolation factor. The data is interpolation by a factor n before the smoothing step.
    window : int, (default : 1)
        Smoothing window that is used to create the convolution and gradually smoothen the line.
    verbose : int [1-5], default: 3
        Print information to screen. A higher number will print more.

    Returns
    -------
    xnew : array-like
        Data points for the x-axis.
    ynew : array-like
        Data points for the y-axis.

    """
    if window is not None:
        if verbose>=3: print('[smoothline] >Smoothing by interpolation..')
        # Specify number of points to interpolate the data
        # Interpolate
        extpoints = np.linspace(0, len(xs), len(xs) * interpol)
        spl = make_interp_spline(range(0, len(xs)), xs, k=3)
        # Compute x-labels
        xnew = spl(extpoints)
        xnew[window:-window]

        # First smoothing on the raw input data
        ynew=None
        if ys is not None:
            ys = _smooth(ys,window)
            # Interpolate ys line
            spl = make_interp_spline(range(0, len(ys)), ys, k=3)
            ynew = spl(extpoints)
            ynew[window:-window]
    else:
        xnew, ynew = xs, ys
    return xnew, ynew


def _smooth(X, window):
    box = np.ones(window) / window
    X_smooth = np.convolve(X, box, mode='same')
    return X_smooth
