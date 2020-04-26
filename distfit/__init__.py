from distfit.distfit import (
    dist,
)
# from distfit.hypotesting import (
    # proba_emperical,
    # proba_parametric,
# )

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.0.0'


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

    out = dist.proba_parametric(y=[-5,1,2,3,10], model=model)


References
----------
    https://github.com/erdogant/distfit

"""
