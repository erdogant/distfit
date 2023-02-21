Generate samples
'''''''''''''''''

With the fitted model it allows to generate samples which can be easily done with the ``generate`` function.

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




.. include:: add_bottom.add