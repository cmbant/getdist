#!/usr/bin/env python
"""
Simple script to build a Mac app bundle for GetDist GUI using PyInstaller.
Assumes uv is installed and available in the PATH.

Usage:
    python scripts/build_mac_app.py [--output-dir OUTPUT_DIR] [--project-dir PROJECT_DIR]
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile


def find_version():
    """Extract version from getdist/__init__.py"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    version_file = open(os.path.join(repo_root, "getdist", "__init__.py")).read()
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
        subprocess.check_call(["uv", "venv", project_dir])

        # Install packages directly with uv pip
        print("Installing PyInstaller and PySide6")
        subprocess.check_call(
            ["uv", "pip", "install", "--python", os.path.join(project_dir, "bin", "python"), "PyInstaller", "PySide6"]
        )

        # Install getdist from the current directory
        print("Installing getdist from local repository")
        subprocess.check_call(
            ["uv", "pip", "install", "--python", os.path.join(project_dir, "bin", "python"), "-e", repo_root]
        )

        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

    return {"venv_dir": project_dir}


def build_mac_app(output_dir, version, env_info):
    """Build the Mac app bundle using PyInstaller"""
    # Detect architecture
    is_arm = platform.machine() == "arm64"
    arch_type = "ARM" if is_arm else "Intel"

    print(f"Building Mac app bundle for GetDist GUI v{version} on {arch_type} architecture...")

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

    # Set target_arch based on the current architecture
    target_arch_value = "'arm64'" if is_arm else "'x86_64'"

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
    target_arch={target_arch_value},
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
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'LSMinimumSystemVersion': '10.13.0',
        'LSApplicationCategoryType': 'public.app-category.developer-tools',
        'LSArchitecturePriority': ['arm64', 'x86_64'] if {is_arm} else ['x86_64'],
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
                spec_path,
            ]
        )

    # Clean up
    shutil.rmtree(temp_dir)

    # Fix Qt frameworks to ensure proper bundle structure
    app_path = os.path.join(output_dir, "GetDist GUI.app")
    print(f"Fixing Qt frameworks in {app_path}...")

    # Run the fix_qt_frameworks.sh script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fix_script = os.path.join(script_dir, "fix_qt_frameworks.sh")

    # Make the script executable
    os.chmod(fix_script, 0o755)

    # Run the script
    try:
        subprocess.check_call([fix_script, app_path])
        print("Qt frameworks fixed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to fix Qt frameworks: {e}")
        print("The app may still work, but signing might fail")

    # Additional fixes for framework structure
    frameworks_dir = os.path.join(app_path, "Contents", "Frameworks")
    if os.path.exists(frameworks_dir):
        print("Applying additional framework structure fixes...")

        # Find all Qt frameworks
        for root, dirs, _ in os.walk(frameworks_dir):
            for dir_name in dirs:
                if dir_name.endswith(".framework"):
                    framework_path = os.path.join(root, dir_name)
                    framework_name = os.path.basename(framework_path).replace(".framework", "")

                    # Create Info.plist if missing
                    resources_dir = os.path.join(framework_path, "Resources")
                    versions_dir = os.path.join(framework_path, "Versions")

                    # Check if this is a versioned framework
                    if os.path.exists(versions_dir):
                        # Find the current version (usually 'A' or '5' for Qt)
                        version_dirs = [
                            d
                            for d in os.listdir(versions_dir)
                            if os.path.isdir(os.path.join(versions_dir, d)) and d != "Current"
                        ]

                        if version_dirs:
                            current_version = version_dirs[0]
                            current_dir = os.path.join(versions_dir, current_version)

                            # Create symlink to Current if missing
                            current_symlink = os.path.join(versions_dir, "Current")
                            if not os.path.exists(current_symlink):
                                print(f"  Creating 'Current' symlink in {framework_path}")
                                os.symlink(current_version, current_symlink)

                            # Create Resources directory if missing
                            version_resources = os.path.join(current_dir, "Resources")
                            if not os.path.exists(version_resources):
                                os.makedirs(version_resources, exist_ok=True)

                            # Create Info.plist if missing
                            info_plist = os.path.join(version_resources, "Info.plist")
                            if not os.path.exists(info_plist):
                                print(f"  Creating Info.plist for {framework_name}")
                                with open(info_plist, "w", encoding="utf-8") as f:
                                    f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>{framework_name}</string>
    <key>CFBundleIdentifier</key>
    <string>org.qt-project.{framework_name}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
</dict>
</plist>""")

                            # Create symlink to Resources if missing
                            resources_symlink = os.path.join(framework_path, "Resources")
                            if not os.path.exists(resources_symlink):
                                print(f"  Creating 'Resources' symlink in {framework_path}")
                                os.symlink("Versions/Current/Resources", resources_symlink)

                            # Create symlink to the framework binary if missing
                            binary_path = os.path.join(current_dir, framework_name)
                            if os.path.exists(binary_path):
                                binary_symlink = os.path.join(framework_path, framework_name)
                                if not os.path.exists(binary_symlink):
                                    print(f"  Creating binary symlink for {framework_name}")
                                    os.symlink(f"Versions/Current/{framework_name}", binary_symlink)
                    else:
                        # Non-versioned framework
                        if not os.path.exists(resources_dir):
                            os.makedirs(resources_dir, exist_ok=True)

                        # Create Info.plist if missing
                        info_plist = os.path.join(resources_dir, "Info.plist")
                        if not os.path.exists(info_plist):
                            print(f"  Creating Info.plist for {framework_name}")
                            with open(info_plist, "w", encoding="utf-8") as f:
                                f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>{framework_name}</string>
    <key>CFBundleIdentifier</key>
    <string>org.qt-project.{framework_name}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
</dict>
</plist>""")

    print(f"Mac app bundle built successfully at {app_path}")


def main():
    """Main function to parse arguments and build the app"""
    parser = argparse.ArgumentParser(description="Build a Mac app bundle for GetDist GUI")
    parser.add_argument("--output-dir", default="dist", help="Output directory for the app bundle")
    parser.add_argument("--project-dir", default="build_env", help="Directory for the build environment")
    args = parser.parse_args()

    # Check if running on macOS
    if sys.platform != "darwin":
        print("Error: This script must be run on macOS.")
        sys.exit(1)

    # Set up project environment
    env_info = setup_project_environment(args.project_dir)

    # Get GetDist version
    version = find_version()

    # Build the Mac app
    build_mac_app(args.output_dir, version, env_info)


if __name__ == "__main__":
    main()
