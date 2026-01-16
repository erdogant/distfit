.. _generate-synthetic-data-generation-unique:

Synthetic Data (Univariate)
''''''''''''''''''''''''''''''''''''''

With the fitted model it allows to generate synthetic data which can be easily done with the ``generate`` function.

Lets see how generating samples works:

.. code:: python

    # import library
    from distfit import distfit
    
    # Generate random normal distributed data
    X = np.random.normal(0, 2, 10000)

    # Initialize model
    dfit = distfit()
    
    # Fit distribution on data X
    dfit.fit_transform(X)
    
    # The fitted distribution can now be used to generate new samples.
    Xgenerate = dfit.generate(n=1000)


Synthetic Data (Multivariate)
'''''''''''''''''''''''''''''''''''''''

.. code:: python

    # import library
    from distfit import distfit
    
    # Generate random normal distributed data
    rng = np.random.default_rng(42)
    mean = [0, 0]
    cov = [[1, 0.6],
           [0.6, 1]]
    X = rng.multivariate_normal(mean, cov, size=2000)
    
    
    # Initialize model
    dfit = distfit(multivariate=True)
    
    # Fit distribution on data X
    dfit.fit_transform(X)
    
    # The fitted distribution can now be used to generate new samples.
    Xgenerate = dfit.generate(n=1000)


    array([[-1.34065007, -0.71402875],
           [-1.02143281, -0.11668787],
           [-1.34764167, -0.7042604 ],
           ...,
           [-0.52385614, -0.62469689],
           [-0.43946412,  0.19682527],
           [ 1.22849301,  0.11507201]], shape=(1000, 2))
           


.. include:: add_bottom.add