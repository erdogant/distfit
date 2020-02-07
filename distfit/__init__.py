from distfit.distfit import (
    fit,
    plot,
    plot_summary,
)
from distfit.hypotesting import (
    proba_emperical,
    proba_parametric,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.4'


# module level doc-string
__doc__ = """
distfit
=====================================================================

Description
-----------
    Probability density function fitting and hypothesis testing.
    Computes best fit to the input emperical distribution for 89 theoretical
    distributions using the Sum of Squared errors (SSE) estimates.

Example
-------
    import distfit as dist

    model = dist.fit(X)

    fig,ax = dist.plot(model)

References
----------
    https://github.com/erdogant/distfit

    See README.md file for more information.

"""
