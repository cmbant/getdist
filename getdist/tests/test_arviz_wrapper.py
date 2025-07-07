import unittest

import numpy as np

try:
    import pymc as pm

    from getdist.arviz_wrapper import arviz_to_mcsamples

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


@unittest.skipUnless(PYMC_AVAILABLE, "PyMC and ArviZ not available")
class TestArvizWrapper(unittest.TestCase):
    """Test the arviz_wrapper functionality using PyMC NUTs example"""

    def setUp(self):
        """Set up test data using PyMC NUTs example similar to the documentation"""

        # Create a simple model similar to the PyMC documentation example
        with pm.Model() as self.model:
            # Create a 3-dimensional normal distribution for faster testing
            self.mu1 = pm.Normal("mu1", mu=0, sigma=1, shape=3)

        # Sample from the model
        with self.model:
            step = pm.NUTS()
            # Use small sample size for faster testing
            self.idata = pm.sample(200, tune=100, init=None, step=step, chains=2, random_seed=42, progressbar=False)

    def test_basic_conversion(self):
        """Test basic conversion from ArviZ InferenceData to MCSamples"""
        # Convert to MCSamples
        mcsamples = arviz_to_mcsamples(self.idata)

        # Check that we have samples
        self.assertGreater(mcsamples.numrows, 0, "Should have samples")

        # Check that we have the right number of samples (2 chains * 200 draws)
        self.assertEqual(mcsamples.numrows, 400, "Should have 400 total samples")

    def test_mean_variance_comparison(self):
        """Test that getdist mean and variance match ArviZ sample_stats expectations"""
        # Convert to MCSamples
        mcsamples = arviz_to_mcsamples(self.idata)

        # Check that we have the right number of parameters

        self.assertEqual(len(mcsamples.paramNames.names), 3, "Should have 3 parameters")
        self.assertEqual(mcsamples.numrows, 400, "Should have 400 total samples")

        # Get means and variances from getdist
        getdist_means = []
        getdist_vars = []

        param_names = mcsamples.paramNames.list()
        for i in range(len(param_names)):
            # Use the MCSamples mean and var methods with parameter index
            mean_val = mcsamples.mean(i)
            var_val = mcsamples.var(i)
            getdist_means.append(mean_val)
            getdist_vars.append(var_val)

        # Get means and variances directly from ArviZ posterior
        arviz_means = []
        arviz_vars = []

        for i in range(3):
            # Get samples for this parameter across all chains and draws
            param_samples = self.idata.posterior.mu1.values[:, :, i].flatten()
            arviz_means.append(np.mean(param_samples))
            arviz_vars.append(np.var(param_samples))

        # Compare means (should be close to 0 since we sampled from Normal(0,1))
        for i in range(3):
            self.assertAlmostEqual(
                getdist_means[i],
                arviz_means[i],
                places=6,
                msg=f"Mean for parameter {i} should match between getdist and arviz",
            )

            # Check that means are reasonably close to 0 (within 3 standard errors)
            std_error = np.sqrt(arviz_vars[i] / 400)  # Standard error of the mean
            self.assertLess(abs(getdist_means[i]), 3 * std_error, msg=f"Mean for parameter {i} should be close to 0")

        # Compare variances (should be close to 1 since we sampled from Normal(0,1))
        for i in range(3):
            self.assertAlmostEqual(
                getdist_vars[i],
                arviz_vars[i],
                places=6,
                msg=f"Variance for parameter {i} should match between getdist and arviz",
            )

    def test_custom_labels(self):
        """Test conversion with custom parameter labels"""
        custom_labels = {f"mu1_{i}": f"\\mu_{{{i}}}" for i in range(3)}

        mcsamples = arviz_to_mcsamples(self.idata, custom_labels=custom_labels)

        # Check that custom labels are applied
        for i, param_info in enumerate(mcsamples.paramNames.names):
            expected_label = f"\\mu_{{{i}}}"
            self.assertEqual(param_info.label, expected_label, f"Custom label for parameter {i} should be applied")


if __name__ == "__main__":
    unittest.main()
