.. _code_directive:

-------------------------------------

Performance
'''''''''''
The performance of ``distfit`` can be examined by various aspects. In this section we will evaluate the detected parameters, and the goodness of fit of the detected probability density function (pdf). 


Parameter fitting
-----------------
Lets evalute the performance of ``distfit`` of the detected parameters when we draw random samples from a normal (Gaussian) distribution with *mu*=0 and *std*=2. We would expect to find *mu* and *std* very close to the input values.

.. code:: python

    from distfit import distfit
    import numpy as np
    # Initialize model and specify distribution to be normal
    X = np.random.normal(0, 2, 5000)

For demonstration puprposes we pre-specify the ``normal`` distribution to find the best parameters. When we do that, as shown below, a *mean* or *loc* of **0.004** is detected and a standard deviation (scale) with **2.02** which is very close to our input parameters. 

.. code:: python

    dist = distfit(distr='norm')
    dist.fit_transform(X)
    print(dist.model)

    # {'distr': <scipy.stats._continuous_distns.norm_gen at 0x15d8406b208>,
    #  'params': (0.00444619012906402, 2.0209991080448138),
    #  'name': 'norm',
    #  'RSS': 0.0021541850376229827,
    #  'loc': 0.00444619012906402,
    #  'scale': 2.0209991080448138,
    #  'arg': (),
    #  'CII_min_alpha': -3.319801522804139,
    #  'CII_max_alpha': 3.328693903062266}

.. code:: python

    dist.plot()

.. _gaus_mu_0:

.. figure:: ../figs/gaus_mu_0.png
    :scale: 80%


Probability Density Function fitting
-------------------------------------

To measure the goodness of fit of *pdfs*, we will evaluate multiple *pdfs* using the **RSS** metrics. The goodness of fit scores are stored in ``dist.summary``. In this example, we will **not** specify any distribution but only provide the emperical data to the model. 

.. code:: python

    dist = distfit()
    dist.fit_transform(X)
    print(dist.summary)

    # 	distr         RSS  ...        scale                                     arg
    # 0        norm  0.00215419  ...        2.021                                      ()
    # 1           t  0.00215429  ...      2.02105                    (2734197.302263666,)
    # 2       gamma  0.00216592  ...   0.00599666                   (113584.76147029496,)
    # 3        beta  0.00220002  ...      39.4803  (46.39522231565038, 47.98055489856441)
    # 4     lognorm  0.00226011  ...      139.173                 (0.014515926633415211,)
    # 5  genextreme  0.00370334  ...      2.01326                   (0.2516817342848604,)
    # 6    dweibull  0.00617939  ...        1.732                   (1.2681369071313497,)
    # 7     uniform    0.244839  ...      14.3579                                      ()
    # 8      pareto    0.358765  ...  2.40844e+08                   (31772216.567824945,)
    # 9       expon    0.360553  ...      7.51848                                      ()

The model detected ``normal`` as the **best** pdf but a good RSS score is also detected for the *t* and *gamma* distribution. But this is not unexpected to see. A summary plot of the evaluated pdfs looks a following:

.. code:: python

    dist.plot_summary()

.. _gaus_mu_0_summary:

.. figure:: ../figs/gaus_mu_0_summary.png
    :scale: 80%


Varying sample size
--------------------
The goodness of fit will change according the number of samples that is provided. In the example above we specified 5000 samples which gave good results. However, with a relative low number of samples, a poor fit can occur. For demonstration purposes we will vary the number of samples and store the *mu*, *std* and detected distribution name.


.. code:: python

    # Create random data with varying number of samples
    samples = np.arange(250, 10000, 250)

    # Initialize model
    distr='norm'
    
    # Estimate parameters for the number of samples
    for s in samples:
        print(s)
        X = np.random.normal(0, 2, s)
        dist.fit_transform(X, verbose=0)
        out.append([dist.model['loc'], dist.model['scale'], dist.model['name'], s])

When we plot the results, ``distfit`` nicely shows that by increasing the number of samples results in a better fit of the parameters. A convergence towards mu=2 and std=0 is clearly seen.


.. |fig1| image:: ../figs/perf_sampling.png
    :scale: 90%

.. |fig2| image:: ../figs/perf_sampling_std.png
    :scale: 90%

.. table:: Sampling example
   :align: center

   +---------+---------+
   | |fig1|  | |fig2|  |
   +---------+---------+



Smoothing window
----------------
If the number of samples is very low, it can be difficult to get a good fit on your data.
A solution is to play with the ``bin`` size, eg. increase bin size. 
Another manner is by smoothing the histogram with the ``smooth`` parameter. The default is set to ``None``.
Lets evaluate the effect of this parameter.

.. code:: python

    # Generate data
    X = np.random.normal(0, 2, 100)

.. code:: python

    # Fit model without smoothing
    model = distfit()
    model.fit_transform(X)
    model.plot()

    # Fit model with heavy smoothing
    model = distfit(smooth=10)
    model.fit_transform(X)
    model.plot()


.. |logo1| image:: ../figs/gaus_mu_0_100samples.png
    :scale: 60%

.. |logo2| image:: ../figs/gaus_mu_0_100samples_smooth10.png
    :scale: 60%

.. table:: Comparison smoothing parameter
   :align: center

   +---------+---------+
   | |logo1| | |logo2| |
   +---------+---------+


Here we are going to combine the number of samples with the smoothing parameter.
It is interesting to see that there is no clear contribution of the smoothing. The legends depicts the smoothing window with the average *mu*.

.. _perf_sampling_mu_smoothing:

.. figure:: ../figs/perf_sampling_mu_smoothing.png
    :scale: 70%


