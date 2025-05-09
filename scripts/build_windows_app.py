#!/usr/bin/env python
"""
Script to build a Windows executable for GetDist GUI using PyInstaller.
Assumes uv is installed and available in the PATH.

Usage:
    python scripts/build_windows_app.py [--output-dir OUTPUT_DIR] [--project-dir PROJECT_DIR]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import re


def find_version():
    """Find the version of GetDist from the package"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    init_file = os.path.join(repo_root, "getdist", "__init__.py")

    with open(init_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', line)
            if match:
                return match.group(1)

    return "0.0.0"


def setup_project_environment(project_dir):
    """Set up a virtual environment with all required dependencies"""
    print(f"Setting up project environment in {project_dir}...")

    # Get the repository root directory
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
        python_path = os.path.join(project_dir, "Scripts", "python.exe")
        subprocess.check_call([
            "uv", "pip", "install",
            "--python", python_path,
            "PyInstaller", "PySide6"
        ])

        # Install getdist from the current directory
        print("Installing getdist from local repository")
        subprocess.check_call([
            "uv", "pip", "install",
            "--python", python_path,
            "-e", repo_root
        ])

        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

    return {
        "venv_dir": project_dir,
        "python_path": python_path
    }


def build_windows_app(output_dir, version, env_info):
    """Build the Windows executable using PyInstaller"""
    print(f"Building Windows executable for GetDist GUI v{version}...")

    # Create a temporary directory for build files
    temp_dir = tempfile.mkdtemp()

    # Get the path to the icon
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    icon_path = os.path.join(repo_root, "getdist", "gui", "images", "Icon.ico")

    # Verify the icon exists
    if not os.path.exists(icon_path):
        print(f"Warning: Icon file not found at {icon_path}")
        # Fall back to PNG if ICO doesn't exist
        icon_path = os.path.join(repo_root, "getdist", "gui", "images", "Icon.png")

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
    target_arch=None,
    codesign_identity=None,
    icon='{icon_path}',
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
"""

    # Write spec file
    spec_path = os.path.join(temp_dir, "GetDistGUI.spec")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)

    # Run PyInstaller using the Python from the virtual environment
    venv_dir = env_info["venv_dir"]
    python_path = env_info["python_path"]

    # Check if pyinstaller is directly accessible in Scripts directory
    pyinstaller_path = os.path.join(venv_dir, "Scripts", "pyinstaller.exe")
    if not os.path.exists(pyinstaller_path):
        print("PyInstaller not found in Scripts directory, using module invocation")
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

    print(f"Windows executable built successfully in {output_dir}\\GetDistGUI")

    # Create a zip file of the GetDistGUI directory
    zip_path = os.path.join(output_dir, f"GetDist-GUI-{version}.zip")
    print(f"Creating zip file at {zip_path}...")

    shutil.make_archive(
        os.path.join(output_dir, f"GetDist-GUI-{version}"),
        'zip',
        output_dir,
        'GetDistGUI'
    )

    print(f"Zip file created at {zip_path}")

    # Create MSI installer if WiX is available
    try:
        print("Attempting to create MSI installer...")
        msi_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_windows_msi.py")
        subprocess.check_call([
            sys.executable,
            msi_script,
            "--input-dir", os.path.join(output_dir, "GetDistGUI"),
            "--output-dir", output_dir,
            "--version", version
        ])
        msi_path = os.path.join(output_dir, f"GetDist-GUI-{version}.msi")
        print(f"MSI installer created at {msi_path}")
        return zip_path, msi_path
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to create MSI installer: {e}")
        print("Continuing with zip file only.")
        return zip_path, None


def main():
    """Main function to parse arguments and build the app"""
    parser = argparse.ArgumentParser(description="Build a Windows executable for GetDist GUI")
    parser.add_argument("--output-dir", default="dist", help="Output directory for the executable")
    parser.add_argument("--project-dir", default="build_env", help="Directory for the build environment")
    args = parser.parse_args()

    # Check if running on Windows
    if sys.platform != "win32":
        print("Error: This script must be run on Windows.")
        sys.exit(1)

    # Set up project environment
    env_info = setup_project_environment(args.project_dir)

    # Get GetDist version
    version = find_version()

    # Build the Windows app
    build_windows_app(args.output_dir, version, env_info)


if __name__ == "__main__":
    main()
