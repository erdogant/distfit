.. _code_directive:

-------------------------------------

Examples
''''''''''

Learn new model with gridsearch and train-test set
--------------------------------------------------

AAA

.. code:: python

    # Import library
    import distfit

    # Load example data set    
    X,y_true = distfit.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = distfit.fit_transform(X, y_true, pos_label='bad', train_test=True, gridsearch=True)

    # The test error will be shown
    results = distfit.plot(model)


Learn new model on the entire data set
--------------------------------------------------

BBBB


.. code:: python

    # Import library
    import distfit

    # Load example data set    
    X,y_true = distfit.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = distfit.fit_transform(X, y_true, pos_label='bad', train_test=False, gridsearch=True)

    # The train error will be shown. Such results are heavily biased as the model also learned on this set of data
    results = distfit.plot(model)

