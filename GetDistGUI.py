#!/usr/bin/env python

from __future__ import absolute_import
import sys

try:
    import getdist
except ImportError:
    import os

    sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
    import getdist

from getdist.gui.mainwindow import run_gui

if sys.platform == "darwin":
    # On Mac need to run .app with pList to get menu name right (and avoid menu bugs)
    import subprocess
    import os

    path = os.path.join(os.path.dirname(getdist.gui.__file__), 'GetDistGUI.app')
    if os.path.exists(path):
        subprocess.call(["/usr/bin/open", "-a", path], env=os.environ)
    else:
        print('GetDistGUI.app not found; package not installed or no valid PySide/PySide2')
        run_gui()
else:
    run_gui()
