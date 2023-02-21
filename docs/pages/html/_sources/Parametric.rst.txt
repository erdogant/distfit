Parametric
'''''''''''

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
    dfit = distfit(distr='norm')
    # Set multiple distributions to test for
    dfit = distfit(distr=['norm','t'])

The ``popular`` set of PDFs contains the following set of distributions and can be used as depicted below:

	+------------+------------+
	| norm       | genextreme | 
	+------------+------------+ 
	| expon      | gamma      | 
	+------------+------------+ 
	| pareto     | lognorm    | 
	+------------+------------+ 
	| dweibull   | beta       | 
	+------------+------------+ 
	| t          | uniform    | 
	+------------+------------+ 

.. code:: python

    # Initialize model and select popular distributions
    dfit = distfit(distr='popular')

 
The ``full`` set contains the following set of distributions:

	+------------+---------------+------------+---------------+--------------+  
	| alpha      | betaprime     | chi2       | expon         | fatiguelife  |  
	+------------+---------------+------------+---------------+--------------+  
	| anglit     | bradford      | cosine     | exponnorm     | fisk         |  
	+------------+---------------+------------+---------------+--------------+  
	| arcsine    | burr          | dgamma     | exponweib     | foldcauchy   |  
	+------------+---------------+------------+---------------+--------------+  
	| arcsine    | cauchy        | dweibull   | exponpow      | foldnorm     |  
	+------------+---------------+------------+---------------+--------------+  
	| beta       | chi           | erlang     | f             | frechet_r[x] |  
	+------------+---------------+------------+---------------+--------------+  
	|gilbrat     | gompertz      | gumbel_r   | gumbel_l      | halfcauchy   |
	+------------+---------------+------------+---------------+--------------+  
	| halfgennorm| hypsecant     | invgamma   | invgauss      | invweibull   |
	+------------+---------------+------------+---------------+--------------+  
	| laplace    | levy          | levy_l [X] | levy_stable[X]| logistic     |
	+------------+---------------+------------+---------------+--------------+  
	+ lognorm    | lomax         | maxwell    | mielke        | nakagami     |
	+------------+---------------+------------+---------------+--------------+  
	| pearson3   | powerlaw      |powerlognorm| powernorm     | rdist        |
	+------------+---------------+------------+---------------+--------------+  
	| rice       | recipinvgauss |semicircular| t             | triang       |
	+------------+---------------+------------+---------------+--------------+  
	|tukeylambda | uniform       | vonmises   | vonmises_line | wald         |
	+------------+---------------+------------+---------------+--------------+  
	| wrapcauchy | gengamma      |genlogistic | frechet_l[x]  | halfnorm     |
	+------------+---------------+------------+---------------+--------------+  
	| genexpon   | genextreme    | gennorm    | gausshyper    | genpareto    | 
	+------------+---------------+------------+---------------+--------------+
	| gamma      |genhalflogistic|halflogistic| johnsonsb     | johnsonsu    |
	+------------+---------------+------------+---------------+--------------+
	| loggamma   | loglaplace    | norm       | pareto        | rayleigh     |
	+------------+---------------+------------+---------------+--------------+
	| reciprocal | truncexpon    | truncnorm  | weibull_min   | weibull_max  |
	+------------+---------------+------------+---------------+--------------+

Note that levy_l and  levy_stable are removed from the full list because it is too slow.
The distributions frechet_r and frechet_l are also not supported anymore.

.. code:: python

    # Initialize model and select all distributions
    dfit = distfit(distr='full')


Residual Sum of Squares (RSS)
-----------------------------
The *RSS* describes the deviation predicted from actual empirical values of data. Or in other words, the differences in the estimates. It is a measure of the discrepancy between the data and an estimation model. A small RSS indicates a tight fit of the model to the data. RSS is computed by:

.. figure:: ../figs/RSS.svg

Where **yi** is the ith value of the variable to be predicted, **xi** is the i-th value of the explanatory variable, and **f(xi)** is the predicted value of **yi** (also termed **y-hat**).


Goodness-of-fit
---------------
Besides *RSS*, there are various other approaches to determine the goodness-of-fit, such as the maximum likelihood estimation (mle), moment matching estimation (mme), quantile matching estimation (qme) or maximizing goodness-of-fit estimation (mge). ``distfit`` may be extended with more approaches in future versions.



Probabilities and multiple test correction
-------------------------------------------

The ``predict`` function: :func:`distfit.distfit.distfit.predict` will compute the probability of samples in the fitted *PDF*. 
Each probability will by default be corrected for multiple testing. Multiple testing correction refers to re-calculating probabilities obtained from a statistical test which was repeated multiple times. In order to retain a prescribed family-wise error rate alpha in an analysis involving more than one comparison, the error rate for each comparison must be more stringent than alpha.
Note that, due to multiple testing approaches, it can occur that samples can be located outside the confidence interval but not marked as significant. See section Algorithm -> Multiple testing for more information.

The following output variables are available. More information can be found under **return** in the docstring.

dfit.predict
	* dfit.results['y_proba']
	* dfit.results['y_pred']
	* dfit.results['df']
	* dfit.summary

The output variable ``y_proba`` is by default corrected for multiple testing using the false discovery rate (fdr).
FDR-controlling procedures are designed to control the expected proportion of "discoveries" that are false.
If desired, other multiple test methods can be choosen, each with its own properties.

.. code:: python

    # Initialize
    dfit = distfit(multtest='holm', alpha=0.01)


+----------------+---------------------------------------------------+
| None           | No multiple testing                               |
+----------------+---------------------------------------------------+
| bonferroni     | one-step correction                               |
+----------------+---------------------------------------------------+
| sidak          | one-step correction                               |
+----------------+---------------------------------------------------+
| holm-sidak     | step down method using Sidak adjustments          |
+----------------+---------------------------------------------------+
|holm            | step-down method using Bonferroni adjustments     |
+----------------+---------------------------------------------------+
|simes-hochberg  | step-up method  (independent)                     |
+----------------+---------------------------------------------------+
|hommel          | closed method based on Simes tests (non-negative) |
+----------------+---------------------------------------------------+
|fdr_bh          | Benjamini/Hochberg  (non-negative)                |
+----------------+---------------------------------------------------+
|fdr_by          | Benjamini/Yekutieli (negative)                    |
+----------------+---------------------------------------------------+
|fdr_tsbh        | two stage fdr correction (non-negative)           |
+----------------+---------------------------------------------------+
|fdr_tsbky       | two stage fdr correction (non-negative)           |
+----------------+---------------------------------------------------+


Input parameters
-----------------
Various input parameters can be specified at the initialization of ``distfit``.

+-----------------+-----+-----------------------+---------------------------------------------------------------+
| Variable name   | type| Default               | Description                                                   |
+-----------------+-----+-----------------------+---------------------------------------------------------------+
| method          | str | 'parametric'          | Specify the method type: 'parametric', 'empirical'            |
+-----------------+-----+-----------------------+---------------------------------------------------------------+
| alpha           |float| 0.05                  | Significance alpha.                                           |
+-----------------+-----+-----------------------+---------------------------------------------------------------+
| multtest        | str | 'fdr_bh'              | Multiple test correction method                               |
+-----------------+-----+-----------------------+---------------------------------------------------------------+
| bins            | int | 50                    | To determine the empirical historgram                         |
+-----------------+-----+-----------------------+---------------------------------------------------------------+
| bound           | int | 'both'                | Directionality to test for significance                       |
|                 |     |                       | Upper and lowerbounds: 'both'                                 |
|                 |     |                       | Upperbounds: 'up', 'high', 'right'                            |
|                 |     |                       | Lowerbounds: 'down', 'low', 'left'                            |
+-----------------+-----+-----------------------+---------------------------------------------------------------+
| distr           | str | 'popular'             | The (set) of distribution to test.                            |
|                 |     |                       | 'popular', 'full'                                             |
|                 |     |                       | 't' : user specified                                          |
|                 |     |                       | 'norm' : user specified                                       |
|                 |     |                       | etc                                                           |
+-----------------+-----+-----------------------+---------------------------------------------------------------+
| n_perm          | int | 10000                 | Number of permutations to model                               |
|                 |     |                       | null-distribution in case of method is 'empirical'            |
+-----------------+-----+-----------------------+---------------------------------------------------------------+


Output variables
-----------------
There are many output parameters provided by ``distfit``.
It all starts with the initialization:

.. code:: python

    # Initialize model and select popular distributions
    dfit = distfit(alpha=0.01)


The object now returns variables that are set by default, except for the ``alpha`` parameter (nothing else is provided). For more details, see the **returns** in the docstrings at :func:`distfit.distfit.distfit`. In the next step, input-data *X* can be provided:

.. code:: python

    # Initialize model and select popular distributions
    dfit.fit_transform(X)

The object can now be feeded with data *X*, using ``fit`` and ``transform`` function, that will add more output variables to the object.
Instead of using the two functions seperately, it can also be performed with ``fit_transform``: :func:`distfit.distfit.distfit.fit_transform`.

The fit_transform outputs the variables *summary*, *distributions* and *model*

dfit.summary
	The summary of the fits across the distributions.

.. code:: python
    
    print(dfit.summary)
    # 	name         RSS  ...      scale                                      arg
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


**dfit.distributions** is a list containing the extracted pdfs from ``scipy``
	The collected distributions.

**dfit.model** contains information regarding the best scoring pdf:
	* dfit.model['RSS']
	* dfit.model['name']
	* dfit.model['model']
	* dfit.model['params']
	* dfit.model['loc']
	* dfit.model['scale']
	* dfit.model['arg']




.. include:: add_bottom.add