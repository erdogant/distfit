.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import distfit

    # Retrieve URLs of malicous and normal urls:
    X, y = distfit.load_example()

    # Learn model on the data
    model = distfit.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = distfit.plot(model)


Installation
''''''''''''

Create environment
------------------


If desired, install ``distfit`` from an isolated Python environment using conda:

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
   conda env remove --name distfit

   # List all the active environments. distfit should be absent.
   conda env list
