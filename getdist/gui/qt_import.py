import six
import matplotlib
from packaging.version import Version

if six.PY3:
    pyside_version = 2
else:
    pyside_version = 1

if pyside_version == 1:
    matplotlib.use('Qt4Agg')

    try:
        if Version(matplotlib.__version__) < Version("2.2.0"):
            matplotlib.rcParams['backend.qt4'] = 'PySide'

    except ImportError:
        pass

else:
    matplotlib.use('Qt5Agg')
    if Version(matplotlib.__version__) < Version("2.2.0"):
        matplotlib.rcParams['backend.qt5'] = 'PySide2'
