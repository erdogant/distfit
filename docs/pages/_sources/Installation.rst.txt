
Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    from distfit import distfit
    import numpy as np

    X = np.random.normal(0, 2, 1000)
    y = [-8,-6,0,1,2,3,4,5,6]

    # Initialize model
    dfit = distfit()

    # Find best theoretical distribution for empirical data X
    dfit.fit_transform(X)
    dfit.plot()

    # Make prediction
    dfit.predict(y)
    dfit.plot()


Installation
''''''''''''

**Create environment**


If desired, install ``distfit`` in an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_distfit python=3.6
    conda activate env_distfit


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install distfit

    # Install directly from github
    pip install git+https://github.com/erdogant/distfit


Uninstalling
''''''''''''

If you want to remove your ``distfit`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. distfit should be listed.
   conda env list

   # Remove the distfit environment
   conda env remove --name env_distfit

   # List all the active environments. distfit should be absent.
   conda env list




.. include:: add_bottom.add