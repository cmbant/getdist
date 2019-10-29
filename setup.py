#!/usr/bin/env python

from __future__ import absolute_import
import io
import re
import os
import sys
import six

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def find_version():
    version_file = io.open(os.path.join(os.path.dirname(__file__), 'getdist/__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_long_description():
    with open('README.rst') as f:
        lines = f.readlines()
        i = -1
        while '=====' not in lines[i]:
            i -= 1
        return "".join(lines[:i])


cmd_class = {}
install_msg = None
package_data = {'getdist': ['analysis_defaults.ini', 'distparam_template.ini'],
                'getdist.gui': ['images/*.png'],
                'getdist.styles': ['*.paramnames', '*.sty']}

if sys.platform == "darwin":
    # Mac wrapper .app bundle
    try:
        # Just check for errors, and skip if no valid PySide
        if six.PY3:
            from PySide2 import QtCore
        else:
            import PySide
    except ImportError as e:
        print("Cannot load PySide or PySide2 - skipping MacOS GetDist GUI app %s" % e)
    else:
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

        package_data['getdist.gui'] += ['GetDist GUI.app/Contents/Info.plist',
                                        'GetDist GUI.app/Contents/MacOS/*',
                                        'GetDist GUI.app/Contents/Resources/*']
        from setuptools.command.develop import develop
        from setuptools.command.install import install
        from setuptools.command.build_py import build_py
        import subprocess
        from distutils import dir_util

        file_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'getdist/gui')
        app_name = 'GetDist GUI.app'


        def make_app():
            # Put python command into app script so it can be run from spotlight etc.
            dir_util.copy_tree(os.path.join(file_dir, 'mac_app'),
                               os.path.join(file_dir, app_name))
            fname = os.path.join(file_dir, app_name + '/Contents/MacOS/GetDistGUI')
            out = []
            with io.open(fname, 'r') as f:
                for line in f.readlines():
                    if 'python=' in line:
                        out.append('python="%s"' % sys.executable)
                    else:
                        out.append(line.strip())
            with io.open(fname, 'w') as f:
                f.write("\n".join(out))
            subprocess.call('chmod +x "%s"' % fname, shell=True)
            fname = os.path.join(file_dir, app_name + '/Contents/Info.plist')
            with io.open(fname, 'r') as f:
                plist = f.read().replace('1.0.0', find_version())
            with io.open(fname, 'w') as f:
                f.write(plist)


        def clean():
            import shutil
            shutil.rmtree(os.path.join(file_dir, app_name), ignore_errors=True)


        class DevelopCommand(develop):

            def run(self):
                make_app()
                develop.run(self)


        class InstallCommand(install):
            def run(self):
                make_app()
                install.run(self)
                clean()


        class BuildCommand(build_py):
            def run(self):
                make_app()
                build_py.run(self)


        cmd_class = {
            'develop': DevelopCommand,
            'install': InstallCommand,
            'build_py': BuildCommand
        }

setup(name='GetDist',
      version=find_version(),
      description='GetDist Monte Carlo sample analysis, plotting and GUI',
      long_description=get_long_description(),
      author='Antony Lewis',
      url="https://github.com/cmbant/getdist",
      zip_safe=False,
      packages=['getdist', 'getdist.gui', 'getdist.tests', 'getdist.styles', 'paramgrid'],
      entry_points={
          'console_scripts': [
              'getdist=getdist.command_line:getdist_command',
              'getdist-gui=getdist.command_line:getdist_gui',
          ]},
      test_suite='getdist.tests',
      package_data=package_data,
      install_requires=[
          'numpy',
          'matplotlib (>=2.2.0)',
          'six',
          "scipy (>=1.0.0)"],
      # PySide or pyside2 is needed for the GUI
      #  optional (for faster file read)
      # 'pandas (>=0.14.0)'
      cmdclass=cmd_class,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          "Programming Language :: Python :: 2",
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      keywords=['MCMC', 'KDE', 'sample', 'density estimation', 'plot', 'figure']
      )
