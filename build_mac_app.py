#!/usr/bin/env python
"""
Simple script to build a Mac app bundle for GetDist GUI using PyInstaller.
Uses uv package manager in a dedicated project directory for clean builds.

Usage:
    python build_mac_app.py [--output-dir OUTPUT_DIR] [--project-dir PROJECT_DIR]
"""

import os
import sys
import shutil
import subprocess
import argparse
import re
import tempfile
import venv
from pathlib import Path


def find_version():
    """Extract version from getdist/__init__.py"""
    version_file = open(os.path.join(os.path.dirname(__file__), 'getdist', '__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def setup_project_environment(project_dir):
    """Set up a dedicated project environment using uv"""
    print(f"Setting up project environment in {project_dir}...")

    # Create project directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)

    # Create a virtual environment
    venv_dir = os.path.join(project_dir, "venv")
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in {venv_dir}...")
        venv.create(venv_dir, with_pip=True)

    # Determine paths
    if sys.platform == "win32":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")

    # Install uv if not already installed
    try:
        subprocess.check_call([python_path, "-m", "pip", "install", "uv"])
        if sys.platform == "win32":
            uv_path = os.path.join(venv_dir, "Scripts", "uv.exe")
        else:
            uv_path = os.path.join(venv_dir, "bin", "uv")
    except subprocess.CalledProcessError:
        print("Failed to install uv. Please install manually.")
        sys.exit(1)

    # Install dependencies using uv
    print("Installing dependencies with uv...")
    try:
        # Install PyInstaller and PySide6
        subprocess.check_call([uv_path, "pip", "install", "PyInstaller", "PySide6"])

        # Install getdist from the current directory
        subprocess.check_call([uv_path, "pip", "install", "-e", os.path.dirname(__file__)])

        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

    return {
        "python": python_path,
        "uv": uv_path,
        "venv_dir": venv_dir
    }


def build_mac_app(output_dir, version, env_info):
    """Build the Mac app bundle using PyInstaller"""
    print(f"Building Mac app bundle for GetDist GUI v{version}...")

    # Create a temporary directory for build files
    temp_dir = tempfile.mkdtemp()

    # Get the path to the icon
    icns_path = os.path.join(os.path.dirname(__file__), "getdist", "gui", "images", "GetDistGUI.icns")
    if not os.path.exists(icns_path):
        # Fall back to PNG if ICNS doesn't exist
        icns_path = os.path.join(os.path.dirname(__file__), "getdist", "gui", "images", "Icon.png")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create PyInstaller spec file
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['getdist/gui/mainwindow.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('getdist/gui/images/*.png', 'getdist/gui/images'),
        ('getdist/gui/images/*.icns', 'getdist/gui/images'),
        ('getdist/analysis_defaults.ini', 'getdist'),
        ('getdist/distparam_template.ini', 'getdist'),
        ('getdist/styles/*.paramnames', 'getdist/styles'),
        ('getdist/styles/*.sty', 'getdist/styles'),
    ],
    hiddenimports=[
        'getdist',
        'getdist.plots',
        'getdist.gui',
        'getdist.styles',
        'scipy.special.cython_special',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qtagg',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GetDistGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GetDistGUI',
)

app = BUNDLE(
    coll,
    name='GetDist GUI.app',
    icon='{icns_path}',
    bundle_identifier='uk.ac.sussex.getdistgui',
    info_plist={{
        'CFBundleDisplayName': 'GetDist GUI',
        'CFBundleName': 'GetDistGUI',
        'CFBundleIdentifier': 'uk.ac.sussex.getdistgui',
        'CFBundleVersion': '{version}',
        'CFBundleShortVersionString': '{version}',
        'NSHumanReadableCopyright': '© Antony Lewis',
        'NSHighResolutionCapable': True,
    }},
)
"""

    # Write spec file
    spec_path = os.path.join(temp_dir, "GetDistGUI.spec")
    with open(spec_path, "w") as f:
        f.write(spec_content)

    # Run PyInstaller using the project's Python
    python_path = env_info["python"]

    print(f"Running PyInstaller with {python_path}...")
    subprocess.check_call([
        python_path,
        "-m",
        "PyInstaller",
        "--clean",
        "--distpath", output_dir,
        "--workpath", os.path.join(temp_dir, "build"),
        spec_path
    ])

    # Clean up
    shutil.rmtree(temp_dir)

    print(f"Mac app bundle built successfully at {os.path.join(output_dir, 'GetDist GUI.app')}")





def main():
    """Main function to parse arguments and build the app"""
    parser = argparse.ArgumentParser(description="Build a Mac app bundle for GetDist GUI")
    parser.add_argument("--output-dir", default="dist", help="Output directory for the app bundle")
    parser.add_argument("--project-dir", default="build_env", help="Directory for the build environment")
    args = parser.parse_args()

    # Check if running on macOS
    if sys.platform != "darwin":
        print("Warning: This script is designed to run on macOS. Some features may not work correctly.")

    # Set up project environment
    env_info = setup_project_environment(args.project_dir)

    # Get GetDist version
    version = find_version()

    # Build the Mac app
    build_mac_app(args.output_dir, version, env_info)


if __name__ == "__main__":
    main()
