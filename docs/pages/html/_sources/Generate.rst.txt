.. _code_directive:
--------------------

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
    dist = distfit()
    
    # Fit distribution on data X
    dist.fit_transform(X)
    
    # The fitted distribution can now be used to generate new samples.
    Xgenerate = dist.generate(n=1000)



.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

