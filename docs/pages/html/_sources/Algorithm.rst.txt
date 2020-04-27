.. _code_directive:

-------------------------------------

Algorithm
'''''''''

The ``distfit`` library uses the goodness of fit test to determine the best probability distribution to the non-censored data. It works by comparing the observed frequency (**f**) to the expected frequency from the model (**f-hat**), and computing the residual sum of squares (RSS). Note that non-censored data is the *full dataset*, and not having any part deleted or suppressed as that can lead to biases.

With ``distfit`` we can test up to 89 univariate distributions, derived from the ``scipy`` library, for which the best fitted distribution is returned with the loc, scale, arg parameters. 


Distributions
---------------------

The distributions to be tested can be specified at initialization using the ``distr`` parameter. 

Distributions options:
	* Manually specifying one or multiple distribution
	* 'popular' set of distributions
	* 'full' set of distributions



Manually specifying
	Manually specifying can be for one or multiple distributions. See example below how its done.

.. code:: python

    # Load library
    from distfit import distfit
    # Initialize model and test only for normal distribution
    dist = distfit(distr='norm')
    # Set multiple distributions to test for
    dist = distfit(distr=['norm','t'])

Popular set
	* The ``popular`` set contains the following set of distributions and can be used as depicted below:
	* **norm, expon, pareto, dweibull, t, genextreme, gamma, lognorm, beta, uniform**

.. code:: python

    # Initialize model and select popular distributions
    dist = distfit(distr='popular')


Full set
	* The ``full`` set contains the following set of distributions:
	* **alpha, anglit, arcsine, beta, betaprime, bradford, burr, cauchy, chi, chi2, cosine, dgamma, dweibull, erlang, expon, exponnorm, exponweib, exponpow, f, fatiguelife, fisk, foldcauchy, foldnorm, frechet_r, frechet_l, genlogistic, genpareto, gennorm, genexpon, genextreme, gausshyper, gamma, gengamma, genhalflogistic, gilbrat, gompertz, gumbel_r, gumbel_l, halfcauchy, halflogistic, halfnorm, halfgennorm, hypsecant, invgamma, invgauss, invweibull, johnsonsb, johnsonsu, laplace, levy, levy_l, levy_stable, logistic, loggamma, loglaplace, lognorm, lomax, maxwell, mielke, nakagami, norm, pareto, pearson3, powerlaw, powerlognorm, powernorm, rdist, reciprocal, rayleigh, rice, recipinvgauss, semicircular, t, triang, truncexpon, truncnorm, tukeylambda, uniform, vonmises, vonmises_line, wald, weibull_min, weibull_max, wrapcauchy**

.. code:: python

    # Initialize model and select popular distributions
    dist = distfit(distr='full')


Residual Sum of Squares (RSS)
-----------------------------
The *RSS* describes the deviation predicted from actual empirical values of data. Or in other words, the differences in the estimates. It is a measure of the discrepancy between the data and an estimation model. A small RSS indicates a tight fit of the model to the data. RSS is computed by:

.. figure:: ../figs/RSS.svg

Where **yi** is the ith value of the variable to be predicted, **xi** is the i-th value of the explanatory variable, and **f(xi)** is the predicted value of **yi** (also termed **y-hat**).


Goodness-of-fit
---------------
Besides *RSS*, there are various other approaches to determine the goodness-of-fit, such as the maximum likelihood estimation (mle), moment matching estimation (mme), quantile matching estimation (qme) or maximizing goodness-of-fit estimation (mge). ``distfit`` may be extended with more approaches in future versions.
