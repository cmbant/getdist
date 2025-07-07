
Using GetDist with MCMC sampler outputs
=======================================

GetDist has built-in support for `Cobaya <https://cobaya.readthedocs.io/>`_ sampler (as well as generic numpy array/plain text format chain files).

To get getdist samples directly from cobaya chains use, e.g.:

.. code-block:: python

    getdist_samples = mcmc.samples(combined=True, skip_samples=0.33, to_getdist=True)

For chain files (or hierarchy of chain directories) stored on disk you can just pass the `chain_dir` argument to `get_single_plotter` or `get_subplot_plotter`,
then reference chains by their root name string. See Cobaya `examples <https://cobaya.readthedocs.io/en/latest/example.html>`_.

GetDist can also be used to analyze and plot samples from a wide variety of MCMC samplers by loading sample arrays directly or integration with ArviZ,
a Python package for exploratory analysis of Bayesian models.

ArviZ Integration
-----------------

GetDist includes an ``arviz_wrapper`` module that can convert ArviZ InferenceData objects (as produced by various samplers) to MCSamples objects.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import arviz as az
    from getdist.arviz_wrapper import arviz_to_mcsamples

    # Load ArviZ example data
    idata = az.load_arviz_data("centered_eight")

    # Convert to MCSamples
    samples = arviz_to_mcsamples(idata)

    # Create plots
    from getdist import plots
    g = plots.get_single_plotter()
    g.plot_1d(samples, 'mu')


PyMC Integration
----------------

PyMC automatically creates ArviZ InferenceData objects by default, making use with GetDist straightforward.

Example: Eight Schools Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pymc as pm
    import numpy as np
    from getdist.arviz_wrapper import arviz_to_mcsamples
    from getdist import plots

    # Eight schools data
    J = 8
    y = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
    sigma = np.array([15., 10., 16., 11., 9., 11., 10., 18.])

    with pm.Model() as model:
        # Priors
        mu = pm.Normal('mu', mu=0, sigma=5)
        tau = pm.HalfNormal('tau', sigma=5)

        # Non-centered parameterization
        theta_tilde = pm.Normal('theta_tilde', mu=0, sigma=1, shape=J)
        theta = pm.Deterministic('theta', mu + tau * theta_tilde)

        # Likelihood
        obs = pm.Normal('obs', mu=theta, sigma=sigma, observed=y)

        # Sample
        idata = pm.sample(1000, tune=1000, chains=4, random_seed=42)

    # Convert to GetDist
    samples = arviz_to_mcsamples(
        idata,
        var_names=['mu', 'tau'],  # Only include these variables
        custom_ranges={'tau':(0, None)},    # important since tau has sharp prior cut
        custom_labels={'mu': r'\mu', 'tau': r'\tau'},
        dataset_label='Eight Schools Model'
    )

    # Create triangle plot
    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], filled=True)


emcee Integration
-----------------

You can convert emcee chains to GetDist format directly, just flatten the array and use directly,
or load chains being careful with the index order.

.. code-block:: python

    import emcee
    from getdist import MCSamples

    ...
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(....))

    # Run MCMC
    sampler.run_mcmc(pos, 5000, progress=True)

    # Get the chains from emcee
    # emcee chains have shape (nsteps, nwalkers, ndim)
    chain = sampler.get_chain(discard=1000)  # Shape: (nsteps, nwalkers, ndim)
    log_prob = sampler.get_log_prob(discard=1000)  # Shape: (nsteps, nwalkers)

    # Convert to MCSamples using multiple chains
    # Each emcee walker should be treated as a separate chain
    # Convert to list of chains (each walker becomes a chain)
    chain_list = [chain[:, i, :] for i in range(chain.shape[1])]  # List of (nsteps, ndim)
    logprob_list = [log_prob[:, i] for i in range(log_prob.shape[1])]  # List of (nsteps,)

    samples = MCSamples(
        samples=chain_list,  # List of arrays, each walker as separate chain
        loglikes=[-lp for lp in logprob_list],  # List of log-likelihood arrays
        names=['m', 'b', 'log_f'],
        labels=[r'm', r'b', r'\log f'],
        label='Line Fitting with emcee'
    )

.. note::
    **Important**: Do not pass the 3D emcee array directly to MCSamples. GetDist would
    interpret each step as a separate chain rather than each walker, which is incorrect.
    Always convert to a list of walker chains as shown above, or faltten the emcee chain.


ArviZ Options
-------------

Custom Parameter Ranges
~~~~~~~~~~~~~~~~~~~~~~~

You can specify parameter ranges so density estimates correctly account for sharp prior cuts:

.. code-block:: python

    samples = arviz_to_mcsamples(
        idata,
        custom_ranges={
            'mu': (-10, 10),      # Both bounds
            'tau': (0, None),     # Lower bound only
            'sigma': (None, 5)    # Upper bound only
        }
    )

Including Weights and Likelihoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your InferenceData contains sample weights or log-likelihood values:

.. code-block:: python

    samples = arviz_to_mcsamples(
        idata,
        weights_var='sample_weight',    # Variable name for weights
        loglikes_var='log_likelihood'   # Variable name for log-likelihoods
    )

Multi-dimensional Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GetDist automatically handles multi-dimensional parameters by flattening them:

.. code-block:: python

    # If you have a parameter 'theta' with shape (8,)
    # It becomes 'theta_0', 'theta_1', ..., 'theta_7'

    # You can customize the naming:
    samples = arviz_to_mcsamples(
        idata,
        include_coords_in_name=True  # Use coordinate names if available
    )

Burn in
------------------------

1. **Burn-in removal**: Most samplers include burn-in samples. Use getdist's settings={'ignore_rows': x} to ignore the first fraction x of each chain, or remove before passing to getdist.
