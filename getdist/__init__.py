__author__ = 'Antony Lewis'
__version__ = "1.0.0"

from getdist.inifile import IniFile
from getdist.paramnames import ParamInfo, ParamNames
from getdist.chains import WeightedSamples
from getdist.mcsamples import MCSamples, loadMCSamples
import os
import sys


def get_defaults_file(name='analysis_defaults.ini'):
    return os.path.join(os.path.dirname(__file__), name)


def set_logging(log):
    import logging

    logging.basicConfig(level=log)


def get_config():
    config_file = os.environ.get('GETDIST_CONFIG', None)
    if not config_file:
        config_file = os.path.join(os.path.dirname(__file__), 'config.ini')
    if os.path.exists(config_file):
        return IniFile(config_file)
    else:
        return IniFile()


def _get_cache_dir():
    if sys.platform == "darwin":
        tmp = os.path.expanduser('~/Library/Caches')
    else:
        import tempfile
        tmp = tempfile.gettempdir()
    tmp = os.path.join(tmp, 'getdist_cache')
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    return tmp


config_ini = get_config()
default_grid_root = config_ini.string('default_grid_root', '')
output_base_dir = config_ini.string('output_base_dir', '')
cache_dir = config_ini.string('cache_dir', _get_cache_dir())
default_getdist_settings = config_ini.string('default_getdist_settings', get_defaults_file())
distparam_template = config_ini.string('distparam_template', get_defaults_file('distparam_template.ini'))
use_plot_data = False  # for legacy compatibility
default_plot_output = config_ini.string('default_plot_output', 'pdf')
loglevel = config_ini.string('logging', '')
if loglevel:
    set_logging(loglevel)
