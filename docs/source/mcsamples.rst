getdist.mcsamples
==================================

.. note::
   **Important Convention**: In GetDist, the ``loglikes`` parameter and related variables represent
   **-log(posterior)**, not -log(likelihood). The posterior is the product of likelihood and prior.
   This means ``loglikes`` contains the negative logarithm of the full
   posterior probability, including both the likelihood and any prior contributions.

.. automodule:: getdist.mcsamples
   :members: loadMCSamples

.. autoclass:: getdist.mcsamples.MCSamples
   :members:
   :inherited-members:


.. automodule:: getdist.mcsamples
   :members: MCSamplesError, ParamError, SettingError
   :noindex:
