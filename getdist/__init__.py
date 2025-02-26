__author__ = 'Antony Lewis'
__version__ = "1.6.1"
__url__ = "https://getdist.readthedocs.io"

import os
import sys
from getdist.inifile import IniFile
from getdist.paramnames import ParamInfo, ParamNames
from getdist.chains import WeightedSamples
from getdist.mcsamples import MCSamples, loadMCSamples


def get_defaults_file(name='analysis_defaults.ini'):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), name)


def set_logging(log):
    import logging

    logging.basicConfig(level=log)


def get_config():
    config_file = os.environ.get('GETDIST_CONFIG', None)
    if not config_file:
        config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.ini')
    if os.path.exists(config_file):
        return IniFile(config_file)
    else:
        return IniFile()


def _get_cache_dir():
    if sys.platform == "darwin":
        tmp = os.path.expanduser('~/Library/Caches')
    elif sys.platform == "win32":
        import tempfile
        tmp = tempfile.gettempdir()
    else:
        tmp = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    return os.path.join(tmp, 'getdist_cache')


def make_cache_dir():
    try:
        if cache_dir:
            os.makedirs(cache_dir)
    except OSError:
        pass
    return cache_dir if cache_dir and os.path.exists(cache_dir) else None


config_ini = get_config()
default_grid_root = config_ini.string('default_grid_root', '') or None
output_base_dir = config_ini.string('output_base_dir', '')
cache_dir = config_ini.string('cache_dir', _get_cache_dir())
default_getdist_settings = config_ini.string('default_getdist_settings', get_defaults_file())
distparam_template = config_ini.string('distparam_template', get_defaults_file('distparam_template.ini'))
use_plot_data = False  # for legacy compatibility
default_plot_output = config_ini.string('default_plot_output', 'pdf')
loglevel = config_ini.string('logging', '')
if loglevel:
    set_logging(loglevel)
