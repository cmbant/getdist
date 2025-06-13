#!/usr/bin/env python
"""
Script to build a Windows executable for GetDist GUI using PyInstaller.
Assumes uv is installed and available in the PATH.

Usage:
    python scripts/build_windows_app.py [--output-dir OUTPUT_DIR] [--project-dir PROJECT_DIR]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


def find_version():
    """Find the version of GetDist from the package"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    init_file = os.path.join(repo_root, "getdist", "__init__.py")

    with open(init_file, encoding="utf-8") as f:
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

        # Create virtual environment with uv, explicitly using Python 3.10
        print("Creating new virtual environment with Python 3.10")
        subprocess.check_call(["uv", "venv", "--python", "3.10", project_dir])

        # Install packages directly with uv pip
        print("Installing PyInstaller and PySide6")
        python_path = os.path.join(project_dir, "Scripts", "python.exe")
        subprocess.check_call(["uv", "pip", "install", "--python", python_path, "PyInstaller", "PySide6"])

        # Install getdist from the current directory
        print("Installing getdist from local repository")
        subprocess.check_call(["uv", "pip", "install", "--python", python_path, "-e", repo_root])

        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

    return {"venv_dir": project_dir, "python_path": python_path}


def build_windows_app(output_dir, version, env_info):
    """Build the Windows executable using PyInstaller"""
    print(f"Building Windows executable for GetDist GUI v{version}...")

    # Create a temporary directory for build files
    temp_dir = tempfile.mkdtemp()

    # Get the path to the icon
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find the first icon that exists
    icon_path = os.path.join(repo_root, "getdist", "gui", "images", "Icon.ico")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create PyInstaller spec file
    # Fix paths to use proper Windows paths with double backslashes
    main_script = os.path.normpath(os.path.join(repo_root, "getdist", "gui", "mainwindow.py"))

    # Verify the main script exists
    if not os.path.exists(main_script):
        print(f"ERROR: Main script not found at {main_script}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Repository root: {repo_root}")
        print("Listing files in getdist/gui directory:")
        gui_dir = os.path.join(repo_root, "getdist", "gui")
        if os.path.exists(gui_dir):
            for f in os.listdir(gui_dir):
                print(f"  {f}")
        else:
            print(f"Directory {gui_dir} does not exist!")
        sys.exit(1)
    else:
        print(f"Found main script at: {main_script}")

    # Create data entries with proper paths
    images_dir = os.path.join(repo_root, "getdist", "gui", "images")
    styles_dir = os.path.join(repo_root, "getdist", "styles")

    # Create data entries list for PyInstaller
    data_entries = []

    # Add image files individually
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            if file.endswith(".png"):
                src = os.path.join(images_dir, file)
                data_entries.append(f"(r'{src}', 'getdist/gui/images')")

    # Add INI files
    analysis_defaults = os.path.join(repo_root, "getdist", "analysis_defaults.ini")
    if os.path.exists(analysis_defaults):
        data_entries.append(f"(r'{analysis_defaults}', 'getdist')")

    distparam_template = os.path.join(repo_root, "getdist", "distparam_template.ini")
    if os.path.exists(distparam_template):
        data_entries.append(f"(r'{distparam_template}', 'getdist')")

    # Add style files individually
    if os.path.exists(styles_dir):
        for file in os.listdir(styles_dir):
            if file.endswith(".paramnames") or file.endswith(".sty"):
                src = os.path.join(styles_dir, file)
                data_entries.append(f"(r'{src}', 'getdist/styles')")

    # Join all data entries
    data_entries_str = ",\n        ".join(data_entries)

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    [r'{main_script}'],
    pathex=[],
    binaries=[
        # Explicitly include Python DLLs to avoid version conflicts
        (r'{os.path.join(os.path.dirname(sys.executable), "python310.dll")}', '.'),
    ],
    datas=[
        {data_entries_str}
    ],
    hiddenimports=[
        'getdist',
        'getdist.plots',
        'getdist.gui',
        'getdist.styles',
        'scipy.special.cython_special',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qtagg',
        # Add multiprocessing imports to fix the python313.dll conflict
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.managers',
        'multiprocessing.popen_spawn_win32',
        'multiprocessing.popen_fork',
        'multiprocessing.popen_forkserver',
        'multiprocessing.popen_spawn_posix',
        'multiprocessing.synchronize',
        'multiprocessing.heap',
        'multiprocessing.resource_tracker',
        'multiprocessing.spawn',
        'multiprocessing.util',
        'multiprocessing.context',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[r'{os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiprocessing_hook.py")}'],
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
    {"" if not icon_path else f"icon=r'{icon_path}'"},
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
        subprocess.check_call(
            [
                python_path,
                "-m",
                "PyInstaller",
                "--clean",
                "--noconfirm",  # Automatically answer yes to prompts
                "--distpath",
                output_dir,
                "--workpath",
                os.path.join(temp_dir, "build"),
                # Add additional options to ensure correct Python version
                "--log-level",
                "DEBUG",  # More verbose logging
                spec_path,
            ]
        )
    else:
        print(f"Running PyInstaller from environment {venv_dir}...")
        subprocess.check_call(
            [
                python_path,
                pyinstaller_path,
                "--clean",
                "--noconfirm",  # Automatically answer yes to prompts
                "--distpath",
                output_dir,
                "--workpath",
                os.path.join(temp_dir, "build"),
                # Add additional options to ensure correct Python version
                "--log-level",
                "DEBUG",  # More verbose logging
                spec_path,
            ]
        )

    # Clean up
    shutil.rmtree(temp_dir)

    print(f"Windows executable built successfully in {output_dir}\\GetDistGUI")


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
