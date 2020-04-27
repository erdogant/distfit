distfit's documentation!
========================

*distfit* is Python package for probability density functions fitting on emperical datasets.
``distfit``can determine the best fit to the input emperical distribution for 89 theoretical distributions using the Sum of Squared errors (SSE) estimates.
The best fitted theoretical distribution is returned with the loc, scale, arg parameters which can then be used
to compute the probability on new data-points.


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
  :maxdepth: 1
  :caption: Methods

  Algorithm
  Cross validation and hyperparameter tuning
  Performance
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
