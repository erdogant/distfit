Quantiles
'''''''''''

The method **quantile** simply computes the confidence intervals based on the *quantiles*.


.. code:: python

    # Load library
    from distfit import distfit

    # Initialize model with quantile method
    dfit = distfit(method='quantile')

    # Some random data
    X = np.random.normal(10, 3, 2000)
    y = [3,4,5,6,10,11,12,18,20]
    # Compute quantiles based on data
    dfit.fit_transform(X)

    # Some results about the CII
    print(dfit.model['CII_min_alpha'])
    # > 5.024718707579791
    # Some results
    print(dfit.model['CII_max_alpha'])
    # > 15.01373120064936

    # Plot
    dfit.plot()


.. |fig_quantile1| image:: ../figs/quantile_plot.png
    :scale: 70%

.. table:: Distribution fit
   :align: center

   +-----------------+
   | |fig_quantile1| |
   +-----------------+

.. code:: python

    # Make prediction
    dfit.predict(y)
    # Plot
    dfit.plot()


.. |fig_quantile2| image:: ../figs/quantile_plot_predict.png
    :scale: 70%

.. table:: Distribution fit
   :align: center

   +-----------------+
   | |fig_quantile2| |
   +-----------------+





.. include:: add_bottom.add