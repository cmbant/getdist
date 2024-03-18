import re
import os
import sys
import shutil
from setuptools import setup

cmd_class = {}
install_msg = None
package_data = {'getdist': ['analysis_defaults.ini', 'distparam_template.ini'],
                'getdist.gui': ['images/*.png'],
                'getdist.styles': ['*.paramnames', '*.sty']}


def find_version():
    version_file = open(os.path.join(os.path.dirname(__file__), 'getdist', '__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


if sys.platform == "darwin":
    # Mac wrapper .app bundle
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    package_data['getdist.gui'] += ['GetDist GUI.app/Contents/Info.plist',
                                    'GetDist GUI.app/Contents/MacOS/*',
                                    'GetDist GUI.app/Contents/Resources/*']
    from setuptools.command.develop import develop
    from setuptools.command.install import install
    from setuptools.command.build_py import build_py
    import subprocess

    file_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'getdist/gui')
    app_name = 'GetDist GUI.app'


    def make_app():
        # Put python command into app script so it can be run from spotlight etc.
        app_dir = os.path.join(file_dir, app_name)
        if os.path.exists(app_dir):
            shutil.rmtree(app_dir)
        shutil.copytree(os.path.join(file_dir, 'mac_app'), app_dir)
        fname = os.path.join(file_dir, app_name + '/Contents/MacOS/GetDistGUI')
        out = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                if 'python=' in line:
                    out.append('python="%s"' % sys.executable)
                else:
                    out.append(line.strip())
        with open(fname, 'w') as f:
            f.write("\n".join(out))
        subprocess.call('chmod +x "%s"' % fname, shell=True)
        fname = os.path.join(file_dir, app_name + '/Contents/Info.plist')
        with open(fname, 'r') as f:
            plist = f.read().replace('1.0.0', find_version())
        with open(fname, 'w') as f:
            f.write(plist)


    def clean():
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

setup(name="getdist",
      zip_safe=False,
      platforms="any",
      package_data=package_data,
      packages=["getdist", "getdist.gui", "getdist.tests", "getdist.styles"],
      test_suite="getdist.tests",
      cmdclass=cmd_class)
