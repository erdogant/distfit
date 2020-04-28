.. _code_directive:

-------------------------------------

Examples
''''''''''

Fit distribution
--------------------------------------------------

.. code:: python

    # Example data
    X = np.random.normal(10, 3, 2000)
    y = [3,4,5,6,10,11,12,18,20]

    # Initialize
    dist = distfit()
    dist.fit_transform(X)

    # Make prediction
    dist.predict(y)



.. |fig1a| image:: ../figs/example_fig1a.png
    :scale: 80%

.. table:: Distribution fit
   :align: center

   +---------+
   | |fig1a| |
   +---------+


Make predictions
--------------------------------------------------

Make some predictions can with the ``predict`` function. 
Due to multiple testing it can occur that samples are outside the boundary 
of the distribution confidence interval but are not marked as significant.

.. code:: python

    # Example data
    X = np.random.normal(10, 3, 2000)
    y = [3,4,5,6,10,11,12,18,20]

    # Initialize
    dist = distfit(distr='full', alpha=0.01)
    dist.fit_transform(X)

    # Make prediction
    dist.predict(y)

.. |fig1b| image:: ../figs/example_fig1b.png
    :scale: 80%

.. table:: Plot distribution with predictions
   :align: center

   +---------+
   | |fig1b| |
   +---------+
