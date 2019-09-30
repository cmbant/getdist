Analysis settings
==================================

Samples are analysed using various analysis settings. These can be specified from a .ini file or overridden using a dictionary.

Default settings from analysis_defaults.ini:

.. literalinclude:: ../../getdist/analysis_defaults.ini
   :language: ini

You can also change the default analysis settings file by setting the GETDIST_CONFIG environment variable to the location of a config.ini
file, where config.ini contains a default_getdist_settings parameter set to the name of the ini file you want to use instead.
