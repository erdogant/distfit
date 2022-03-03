distfit's documentation!
========================

``distfit`` is a python package for probability density fitting of univariate distributions on non-censored data. With the Residual Sum of Squares (RSS) we determine the best fit across 89 theoretical distributions for which the best fitted theoretical distribution is returned with the loc, scale, arg parameters. The probability of new data-points can then be assed for significance.

.. tip::
	**distfit** vs. **fitdist**
	The ``fitdist`` library is directly build on the ``distfit`` library so there are no changes between the two except for naming. If you desire to use the ``fitdist`` library thats great! Just change all the examples in this documentation into ``fitdist``. Thats it. Have fun!


Content
=======

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
  :caption: Save and Load

  Save and Load


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples


.. toctree::
  :maxdepth: 2
  :caption: Documentation
  
  Documentation
  Coding quality
  distfit.distfit

* :ref:`genindex`


Quick install
---------------

.. code-block:: console

   pip install distfit


Source code and issue tracker
------------------------------

`Github distfit <https://github.com/erdogant/distfit/>`_.
Please report bugs, issues and feature extensions there.


Citing *distfit*
----------------

The bibtex can be found in the right side menu at the `github page <https://github.com/erdogant/distfit/>`_.


Sponsor this project
------------------------------

If you like this project, **star** this repo and become a **sponsor**!
Read more why this is important on my sponsor page!

.. raw:: html

	<iframe src="https://github.com/sponsors/erdogant/button" title="Sponsor erdogant" height="35" width="116" style="border: 0;"></iframe>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


<script src="https://erdogant.github.io/carbon_ads/carbon_ads.js"></script>

