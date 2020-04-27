.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models is desired as the learning proces of a model for ``distfit`` can take up to hours.
In order to accomplish this, we created two functions: function :func:`distfit.save` and function :func:`distfit.load`
Below we illustrate how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`distfit.save`:

.. code:: python

    import distfit

    # Load example data
    X,y_true = distfit.load_example()

    # Learn model
    model = distfit.fit_transform(X, y_true, pos_label='bad')

    Save model
    status = distfit.save(model, 'learned_model_v1')



Loading
----------------------

Loading a learned model can be done using the function :func:`distfit.load`:

.. code:: python

    import distfit

    # Load model
    model = distfit.load(model, 'learned_model_v1')
