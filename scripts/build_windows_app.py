#!/usr/bin/env python
"""
Script to build a Windows executable for GetDist GUI using PyInstaller.
Assumes uv is installed and available in the PATH.

Usage:
    python scripts/build_windows_app.py [--output-dir OUTPUT_DIR] [--project-dir PROJECT_DIR]
"""

import argparse
import json
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


def get_python_runtime_info(python_path):
    """Get interpreter details needed to locate the matching Python DLL."""
    output = subprocess.check_output(
        [
            python_path,
            "-c",
            (
                "import json, sys;"
                "print(json.dumps({"
                "'base_exec_prefix': sys.base_exec_prefix,"
                "'base_prefix': sys.base_prefix,"
                "'executable': sys.executable,"
                "'version_info': [sys.version_info.major, sys.version_info.minor],"
                "}))"
            ),
        ],
        text=True,
    ).strip()
    return json.loads(output)


def get_python_dll_candidates(runtime_info):
    """Return likely locations for the interpreter's Python DLL."""
    major, minor = runtime_info["version_info"]
    dll_name = f"python{major}{minor}.dll"
    candidates = [
        os.path.join(runtime_info["base_exec_prefix"], dll_name),
        os.path.join(runtime_info["base_prefix"], dll_name),
        os.path.join(os.path.dirname(runtime_info["executable"]), dll_name),
    ]

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.normpath(candidate))
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(candidate)

    return unique_candidates


def find_existing_path(candidates):
    """Return the first existing path from a list of candidates."""
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find Python DLL. Checked: {', '.join(candidates)}")


def get_python_dll_path(python_path):
    """Resolve the Python DLL that matches the interpreter used for the build."""
    runtime_info = get_python_runtime_info(python_path)
    return find_existing_path(get_python_dll_candidates(runtime_info))


def get_build_python_selector():
    """Return the interpreter selector used to create the build environment."""
    return os.path.abspath(sys.executable)


def build_pyinstaller_spec(main_script, data_entries_str, python_dll_path, runtime_hook_path, icon_path):
    """Build the PyInstaller spec content."""
    icon_line = "" if not icon_path else f"icon=r'{icon_path}',"
    return f"""# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = [
    'getdist',
    'getdist.plots',
    'getdist.gui',
    'getdist.styles',
    'scipy.special.cython_special',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qtagg',
]
hiddenimports += collect_submodules('multiprocessing')

a = Analysis(
    [r'{main_script}'],
    pathex=[],
    binaries=[
        # Explicitly include the Python DLL used by the build interpreter
        (r'{python_dll_path}', '.'),
    ],
    datas=[
        {data_entries_str}
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[r'{runtime_hook_path}'],
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
    {icon_line}
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

        # Create virtual environment with uv using the invoking Python interpreter
        build_python = get_build_python_selector()
        print(f"Creating new virtual environment with Python from {build_python}")
        subprocess.check_call(["uv", "venv", "--python", build_python, project_dir])

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

    python_path = env_info["python_path"]
    python_dll_path = get_python_dll_path(python_path)
    runtime_hook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiprocessing_hook.py")
    print(f"Using Python DLL from build interpreter: {python_dll_path}")

    spec_content = build_pyinstaller_spec(
        main_script=main_script,
        data_entries_str=data_entries_str,
        python_dll_path=python_dll_path,
        runtime_hook_path=runtime_hook_path,
        icon_path=icon_path,
    )

    # Write spec file
    spec_path = os.path.join(temp_dir, "GetDistGUI.spec")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)

    # Run PyInstaller using the Python from the virtual environment
    venv_dir = env_info["venv_dir"]

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
