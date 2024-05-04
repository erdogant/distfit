from distfit.distfit import distfit

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.8.0'


# module level doc-string
__doc__ = """
distfit
=====================================================================
distfit is a python package for probability density fitting of univariate distributions for random variables.
With the random variable as an input, distfit can find the best fit for parametric, non-parametric, and discrete distributions.

For the parametric approach, the distfit library can determine the best fit across 89 theoretical distributions.
To score the fit, one of the scoring statistics for the good-of-fitness test can be used used, such as RSS/SSE, Wasserstein,
Kolmogorov-Smirnov (KS), or Energy. After finding the best-fitted theoretical distribution, the loc, scale,
and arg parameters are returned, such as mean and standard deviation for normal distribution.

In case of the non-parametric approach, the distfit library contains two methods, the quantile and percentile method.
Both methods assume that the data does not follow a specific probability distribution. In the case of the quantile method,
the quantiles of the data are modeled whereas for the percentile method, the percentiles are modeled.

In case the dataset contains discrete values, the distift library contains the option for discrete fitting.
The best fit is then derived using the binomial distribution.


Example
-------
>>> from distfit import distfit
>>> import numpy as np
>>>
>>> X = np.random.normal(0, 2, 1000)
>>> y = [-8,-6,0,1,2,3,4,5,6]
>>>
>>> dfit = distfit()
>>> results = dfit.fit_transform(X)
>>> dfit.plot()
>>>
>>> # Plot summary
>>> dfit.plot_summary()
>>>
>>> # Make prediction
>>> results_proba = dfit.predict(y)
>>>
>>> # Plot PDF
>>> fig, ax = dfit.plot(chart='pdf', n_top=1)
>>>
>>> # Add the CDF to the plot
>>> fig, ax = dfit.plot(chart='cdf', n_top=1, ax=ax)
>>>
>>> # QQ-plot for top 10 fitted distributions
>>> fig, ax = dfit.qqplot(X, n_top=10)
>>>

References
----------
    * https://erdogant.github.io/distfit

"""
