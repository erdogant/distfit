distfit's documentation!
========================

``distfit`` is a python package for probability density fitting of univariate distributions on non-censored data. With the Residual Sum of Squares (RSS) we determine the best fit across 89 theoretical distributions for which the best fitted theoretical distribution is returned with the loc, scale, arg parameters. The probability of new data-points can then be assed for significance.

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
  :maxdepth: 1
  :caption: Code Documentation
  
  Coding quality
  distfit.distfit



Quick install
-------------

.. code-block:: console

   pip install distfit




Source code and issue tracker
------------------------------

Available on Github, `erdogant/distfit <https://github.com/erdogant/distfit/>`_.
Please report bugs, issues and feature extensions there.

Citing *distfit*
----------------
Here is an example BibTeX entry:

	@misc{erdogant2019distfit,
	  title={distfit},
	  author={Erdogan Taskesen},
	  year={2019},
	  howpublished={\url{https://github.com/erdogant/distfit}}}



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
