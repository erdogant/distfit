.. _code_directive:
-------------------------------------

Algorithm
'''''''''

The ``distfit`` library uses the goodness of fit test to determine the best probability distribution to the non-censored data. It works by comparing the observed frequency (*f*) to the expected frequency from the model (*f-hat*), and computing the residual sum of squares (RSS). Note that non-censored data is the *full dataset*, and not having any part deleted or suppressed as that can lead to biases.

With ``distfit`` we can test up to 89 univariate distributions, derived from the ``scipy`` library, for which the best fitted distribution is returned with the loc, scale, arg parameters. 


Distributions
---------------------

The distributions to be tested can be specified at initialization using the ``distr`` parameter. 
The distributions options are 
1. Manually specifying one or multiple distribution
2. *popular* set of distributions
3. *full* set of distributions

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



Predictions
------------

After making predictions using the predict function: :func:`distfit.distfit.distfit.predict`, the following output variables are available. More information can be found under **return** in the docstring.

**dist.predict**
	* dist.y_proba
	* dist.y_pred
	* dist.df
	* dist.summary


    Praw : list of float
        Pvalues.
    multtest : str, default: 'fdr_bh'
        Multiple testing method. Options are:
            None : No multiple testing
            'bonferroni' : one-step correction
            'sidak' : one-step correction
            'holm-sidak' : step down method using Sidak adjustments
            'holm' : step-down method using Bonferroni adjustments
            'simes-hochberg' : step-up method  (independent)
            'hommel' : closed method based on Simes tests (non-negative)
            'fdr_bh' : Benjamini/Hochberg  (non-negative)
            'fdr_by' : Benjamini/Yekutieli (negative)
            'fdr_tsbh' : two stage fdr correction (non-negative)
            'fdr_tsbky' : two stage fdr correction (non-negative)



Output variables
-----------------
There are many output parameters provided by ``distfit``.
It all starts with the initialization:

.. code:: python

    # Initialize model and select popular distributions
    dist = distfit(alpha=0.01)


The object now returns variables that are set by default, except for the ``alpha`` parameter (nothing else is provided). For more details, see the **returns** in the docstrings at :func:`distfit.distfit.distfit`. In the next step, input-data *X* can be provided:

.. code:: python

    # Initialize model and select popular distributions
    dist.fit_transform(X)

The object can now be feeded with data *X*, using ``fit`` and ``transform`` function, that will add more output variables to the object.
Instead of using the two functions seperately, it can also be performed with ``fit_transform``: :func:`distfit.distfit.distfit.fit_transform`.

The fit_transform outputs the variables *summary*, *distributions* and *model*

dist.summary
	The summary of the fits across the distributions.

.. code:: python
    
    print(dist.summary)
    # 	distr         RSS  ...      scale                                      arg
    # 0       gamma  0.00185211  ...  0.0370159                     (3004.147964288284,)
    # 1           t  0.00186936  ...    2.02883                     (2517332.591227023,)
    # 2        norm  0.00186945  ...    2.02882                                       ()
    # 3        beta  0.00186949  ...    37.7852  (39.068072383763294, 46.06165256503778)
    # 4     lognorm  0.00197359  ...    57.4149                   (0.03537982752374607,)
    # 5  genextreme  0.00297519  ...     2.0106                    (0.2437702978900108,)
    # 6    dweibull  0.00695379  ...    1.73297                    (1.2545534252305621,)
    # 7     uniform    0.241881  ...    14.1011                                       ()
    # 8       expon    0.353202  ...    6.99491                                       ()
    # 9      pareto    0.634924  ...    1.42095                    (0.5384782616155881,)


**dist.distributions** is a list containing the extracted pdfs from ``scripy``
	The collected distributions.

**dist.model** contains information regarding the best scoring pdf:
	* dist.model['RSS']
	* dist.model['name']
	* dist.model['distr']
	* dist.model['params']
	* dist.model['loc']
	* dist.model['scale']
	* dist.model['arg']
