#!/usr/bin/env python
"""
Simple script to build a Mac app bundle for GetDist GUI using PyInstaller.
Assumes uv is installed and available in the PATH.

Usage:
    python scripts/build_mac_app.py [--output-dir OUTPUT_DIR] [--project-dir PROJECT_DIR]
"""

import os
import sys
import shutil
import subprocess
import argparse
import re
import tempfile
from pathlib import Path


def find_version():
    """Extract version from getdist/__init__.py"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    version_file = open(os.path.join(repo_root, 'getdist', '__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def setup_project_environment(project_dir):
    """Set up a dedicated project environment using uv"""
    print(f"Setting up project environment in {project_dir}...")

    # Create project directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)

    # Create a virtual environment and install dependencies using uv
    print("Creating virtual environment and installing dependencies...")

    # Get repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        # Create virtual environment
        subprocess.check_call([
            "uv", "venv", project_dir
        ])

        # Install PyInstaller and PySide6
        subprocess.check_call([
            "uv", "pip", "install", "--project", project_dir, "PyInstaller", "PySide6"
        ])

        # Install getdist from the current directory
        subprocess.check_call([
            "uv", "pip", "install", "--project", project_dir, "-e", repo_root
        ])

        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

    return {
        "venv_dir": project_dir
    }


def build_mac_app(output_dir, version, env_info):
    """Build the Mac app bundle using PyInstaller"""
    print(f"Building Mac app bundle for GetDist GUI v{version}...")

    # Create a temporary directory for build files
    temp_dir = tempfile.mkdtemp()

    # Get the path to the icon
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    icns_path = os.path.join(repo_root, "getdist", "gui", "images", "GetDistGUI.icns")
 
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create PyInstaller spec file
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{os.path.join(repo_root, "getdist/gui/mainwindow.py")}'],
    pathex=[],
    binaries=[],
    datas=[
        ('{os.path.join(repo_root, "getdist/gui/images/*.png")}', 'getdist/gui/images'),
        ('{os.path.join(repo_root, "getdist/gui/images/*.icns")}', 'getdist/gui/images'),
        ('{os.path.join(repo_root, "getdist/analysis_defaults.ini")}', 'getdist'),
        ('{os.path.join(repo_root, "getdist/distparam_template.ini")}', 'getdist'),
        ('{os.path.join(repo_root, "getdist/styles/*.paramnames")}', 'getdist/styles'),
        ('{os.path.join(repo_root, "getdist/styles/*.sty")}', 'getdist/styles'),
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
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)

    # Run PyInstaller using uv run to ensure correct environment
    venv_dir = env_info["venv_dir"]

    print(f"Running PyInstaller with uv in environment {venv_dir}...")
    subprocess.check_call([
        "uv", "run",
        "--project", venv_dir,
        "pyinstaller",
        "--clean",
        "--noconfirm",  # Automatically answer yes to prompts
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
