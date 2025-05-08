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
        # Remove existing environment if it exists
        if os.path.exists(project_dir):
            print(f"Removing existing environment at {project_dir}")
            shutil.rmtree(project_dir)

        # Create virtual environment with uv
        print("Creating new virtual environment")
        subprocess.check_call([
            "uv", "venv", project_dir
        ])

        # Install packages directly with uv pip
        print("Installing PyInstaller and PySide6")
        subprocess.check_call([
            "uv", "pip", "install",
            "--python", os.path.join(project_dir, "bin", "python"),
            "PyInstaller", "PySide6"
        ])

        # Install getdist from the current directory
        print("Installing getdist from local repository")
        subprocess.check_call([
            "uv", "pip", "install",
            "--python", os.path.join(project_dir, "bin", "python"),
            "-e", repo_root
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

    # Create entitlements file for hardened runtime
    entitlements_path = os.path.join(temp_dir, "entitlements.plist")
    entitlements_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
</dict>
</plist>
"""
    with open(entitlements_path, "w", encoding="utf-8") as f:
        f.write(entitlements_content)

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
    entitlements_file='{entitlements_path}',
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

    # Run PyInstaller using the Python from the virtual environment
    venv_dir = env_info["venv_dir"]
    python_path = os.path.join(venv_dir, "bin", "python")

    # Check if pyinstaller is directly accessible in bin directory
    pyinstaller_path = os.path.join(venv_dir, "bin", "pyinstaller")
    if not os.path.exists(pyinstaller_path):
        print("PyInstaller not found in bin directory, using module invocation")
        print(f"Running PyInstaller as module with Python from {venv_dir}...")
        subprocess.check_call([
            python_path,
            "-m", "PyInstaller",
            "--clean",
            "--noconfirm",  # Automatically answer yes to prompts
            "--distpath", output_dir,
            "--workpath", os.path.join(temp_dir, "build"),
            spec_path
        ])
    else:
        print(f"Running PyInstaller from environment {venv_dir}...")
        subprocess.check_call([
            python_path,
            pyinstaller_path,
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
