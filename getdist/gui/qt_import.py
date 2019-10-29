import six
import matplotlib
import sys
import os

pyside_version = 2 if six.PY3 else 1
using_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

matplotlib.use('Qt4Agg' if pyside_version == 1 else 'Qt5Agg')

try:
    if pyside_version == 1:
        import PySide
    else:
        from PySide2 import QtCore
except ImportError as e:
    if pyside_version == 1:
        print("Can't import PySide, install PySide or (better) use PySide 2 with Python 3")
    else:
        if 'DLL load failed' in str(e):
            print('DLL load failed attempting to load PySide2: problem with your python configuration')
        else:
            print(e)
            print("Can't import PySide2 modules, for python 3 you need to install Pyside2")
    if not using_conda:
        print('Using Anaconda is probably the most reliable method')
    print("E.g. make and use a new environment using conda-forge")
    print('conda create -n py37forge -c conda-forge python=3.7 scipy pandas matplotlib PySide2')

    sys.exit(-1)
