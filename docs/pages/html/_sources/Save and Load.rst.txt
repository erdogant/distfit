.. _code_directive:

-------------------------------------

Save and Load
''''''''''''''

Saving and loading models can be desired to start from a previous learning point.
In order to accomplish this, two functions are implemented: function :func:`distfit.save` and function :func:`distfit.load`
Below is an illustration how to save and load models.


Saving
----------------

Saving a learned model can be done using the function :func:`distfit.save`:

.. code:: python

    from distfit import distfit
    import numpy as np
    # Example data
    X = np.random.normal(0, 2, 5000)
    y = [-8,-6,0,1,2,3,4,5,6]
    
    dist = distfit()
    dist.fit_transform(X)
    dist.predict(y)
    
    # Save model
    dist.save('my_first_model.pkl')


Loading
----------------------

Loading a learned model can be done using the function :func:`dist.load`:

.. code:: python

    # Initialize
    dist = distfit()

    # Load model
    dist.load('my_first_model.pkl')
