Percentiles
'''''''''''

The method **percentile** simply computes the confidence intervals based on the *percentiles*.


.. code:: python

    # Load library
    from distfit import distfit

    # Initialize model with percentile method
    dfit = distfit(method='percentile')

    # Some random data
    X = np.random.normal(10, 3, 2000)
    y = [3,4,5,6,10,11,12,18,20]
    # Compute Percentiles based on data
    dfit.fit_transform(X)

    # Some results about the CII
    print(dfit.model['CII_min_alpha'])
    # > 4.0714359161939235
    # Some results
    print(dfit.model['CII_max_alpha'])
    # > 16.00598292777584

    # Plot
    dfit.plot()


.. |fig_percentile1| image:: ../figs/percentile_plot.png
    :scale: 70%

.. table:: Distribution fit
   :align: center

   +-------------------+
   | |fig_percentile1| |
   +-------------------+

.. code:: python

    # Make prediction
    dfit.predict(y)
    # Plot
    dfit.plot()


.. |fig_percentile2| image:: ../figs/percentile_plot_predict.png
    :scale: 70%

.. table:: Distribution fit
   :align: center

   +-------------------+
   | |fig_percentile2| |
   +-------------------+


+-----+-----------+----------+-----+------------+
|   y |   y_proba | y_pred   |   P |   teststat |
+=====+===========+==========+=====+============+
|   3 |         0 | down     |   0 |  -2.60164  |
+-----+-----------+----------+-----+------------+
|   4 |         0 | down     |   0 |  -1.60164  |
+-----+-----------+----------+-----+------------+
|   5 |         1 | none     |   1 |  -0.601636 |
+-----+-----------+----------+-----+------------+
|   6 |         1 | none     |   1 |   0.398364 |
+-----+-----------+----------+-----+------------+
|  10 |         1 | none     |   1 |   4.39836  |
+-----+-----------+----------+-----+------------+
|  11 |         1 | none     |   1 |   5.39836  |
+-----+-----------+----------+-----+------------+
|  12 |         1 | none     |   1 |   6.39836  |
+-----+-----------+----------+-----+------------+
|  18 |         0 | up       |   0 |  12.3984   |
+-----+-----------+----------+-----+------------+
|  20 |         0 | up       |   0 |  14.3984   |
+-----+-----------+----------+-----+------------+





.. include:: add_bottom.add