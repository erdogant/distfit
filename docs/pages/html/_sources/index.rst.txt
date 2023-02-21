distfit's documentation!
========================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |repo-size| |donate|


.. note::
	`Medium Blog: How to Find the Best Theoretical Distribution for Your Data. <https://towardsdatascience.com/how-to-find-the-best-theoretical-distribution-for-your-data-a26e5673b4bd>`_


.. |logo| image:: ../figs/logo.png

.. table::
   :align: center

   +----------+
   | |logo|   |
   +----------+

-----------------------------------



``distfit`` is a python package for probability density fitting of univariate distributions for random variables.
With the random variable as an input, distfit can find the best fit for parametric, non-parametric, and discrete distributions.

* For the parametric approach, the distfit library can determine the best fit across 89 theoretical distributions.
  To score the fit, one of the scoring statistics for the good-of-fitness test can be used used, such as RSS/SSE, Wasserstein,
  Kolmogorov-Smirnov (KS), or Energy. After finding the best-fitted theoretical distribution, the loc, scale,
  and arg parameters are returned, such as mean and standard deviation for normal distribution.

* For the non-parametric approach, the distfit library contains two methods, the quantile and percentile method.
  Both methods assume that the data does not follow a specific probability distribution. In the case of the quantile method,
  the quantiles of the data are modeled whereas for the percentile method, the percentiles are modeled.

* In case the dataset contains discrete values, the distift library contains the option for discrete fitting.
  The best fit is then derived using the binomial distribution.


.. |fig1| image:: ../figs/distfit.png

.. table::
   :align: center

   +----------+
   | |fig1|   |
   +----------+


-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/distfit/pages/html/Documentation.html>`_ in various ways, have a look at the `sponser page <https://erdogant.github.io/distfit/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/distfit/>`_ page.

	.. code-block:: console

	   pip install distfit

-----------------------------------


Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Background
   
   Abstract


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation

.. toctree::
  :maxdepth: 3
  :caption: Methods

  Parametric
  Quantile
  Percentile
  Discrete

.. toctree::
  :maxdepth: 3
  :caption: Generate samples

  Generate

.. toctree::
  :maxdepth: 3
  :caption: Performance

  Performance

.. toctree::
  :maxdepth: 3
  :caption: Plots

  Plots

.. toctree::
  :maxdepth: 3
  :caption: Save and Load

  Save and Load


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples

.. toctree::
  :maxdepth: 2
  :caption: Functions

  Functions


.. toctree::
  :maxdepth: 2
  :caption: Documentation
  
  Documentation
  Coding quality
  distfit.distfit




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/distfit/pages/html/Documentation.html#medium-blog

.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/distfit
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/distfit

.. |stars| image:: https://img.shields.io/github/stars/erdogant/distfit
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/distfit

.. |python| image:: https://img.shields.io/pypi/pyversions/distfit.svg
    :alt: |Python
    :target: https://erdogant.github.io/distfit/

.. |pypi| image:: https://img.shields.io/pypi/v/distfit.svg
    :alt: |Python Version
    :target: https://pypi.org/project/distfit/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/distfit/

.. |LOC| image:: https://sloc.xyz/github/erdogant/distfit/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/distfit

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/distfit?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/distfit

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/distfit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/distfit

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/distfit/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/distfit.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/distfit/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/distfit.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/distfit/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/distfit/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/distfit/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/231843440.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/231843440


.. include:: add_bottom.add
