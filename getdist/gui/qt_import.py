import six
import matplotlib
import sys
import os
from packaging.version import Version

pyside_version = 2 if six.PY3 else 1
using_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

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

if pyside_version == 2:
    try:
        from PySide2 import QtCore
    except ImportError as e:
        if 'DLL load failed' in str(e):
            print('DLL load failed attempting to load PySide2: problem with your python configuration')
            print('Using Anaconda, try to make and use a new environment using conda-forge, e.g.')
        else:
            print(e)
            print("Can't import PySide2 modules, for python 3 you need to install Pyside2")
            if not using_conda:
                print('Using Anaconda is probably the most reliable method')
            print("E.g. make and use a new environment using conda-forge")
        print('conda create -n py37forge -c conda-forge python=3.7 scipy pandas matplotlib packaging PySide2')

        sys.exit(-1)
else:
    try:
        import PySide
    except ImportError:
        print("Can't import PySide modules, install PySide")
        if using_conda:
            print("To install use 'conda install pyside' or 'conda install -c conda-forge pyside'")
        else:
            print(
                "Use 'pip install PySide', or to avoid compile errors install pre-build package using apt get install.")
            print("Alternatively switch to using Anaconda python distribution and get it with that.")
        sys.exit(-1)


