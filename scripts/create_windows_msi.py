#!/usr/bin/env python
"""
Script to create an MSI installer for the GetDist GUI Windows application.
Uses the WiX Toolset to create the installer.

The MSI installer is configured to properly handle upgrades, ensuring that:
1. Previous versions are removed before installing the new version
2. Same version upgrades are allowed without creating duplicates in Add/Remove Programs
3. All features from previous versions are properly removed

Usage:
    python scripts/create_windows_msi.py --input-dir dist/GetDistGUI --output-dir dist --version 1.6.3
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid


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


def check_wix_installed():
    """Check if WiX Toolset is installed"""
    try:
        # Try to run candle.exe to check if WiX is installed
        subprocess.check_output(["where", "candle.exe"], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False


def create_wix_files(input_dir, output_dir, version):
    """Create WiX source files for the installer"""
    print("Creating WiX source files...")

    # Create a temporary directory for WiX files
    temp_dir = tempfile.mkdtemp()

    # Generate a unique product ID
    product_id = str(uuid.uuid4())
    upgrade_code = "61DAB74D-B917-4D5D-B6DA-BBA73C69C159"  # Keep this constant across versions

    # Create WiX XML file
    wxs_path = os.path.join(temp_dir, "GetDistGUI.wxs")

    # Get the icon path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ico_path = os.path.join(repo_root, "getdist", "gui", "images", "Icon.ico")
    if not os.path.exists(ico_path):
        # Fall back to PNG if ICO doesn't exist
        ico_path = os.path.join(repo_root, "getdist", "gui", "images", "Icon.png")
    icon_path = ico_path

    # Create the WiX XML content
    wxs_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
    <Product Id="{product_id}"
             Name="GetDist GUI"
             Language="1033"
             Version="{version}"
             Manufacturer="Antony Lewis"
             UpgradeCode="{upgrade_code}">

        <Package InstallerVersion="200" Compressed="yes" InstallScope="perMachine" />

        <!-- MajorUpgrade element handles removing previous versions before installing the new one
             AllowSameVersionUpgrades="yes" - Allows reinstalling the same version without duplicates in Add/Remove Programs
             RemoveFeatures="ALL" - Ensures all features from previous versions are removed
             Schedule="afterInstallInitialize" - Ensures previous version is removed before new one is installed -->
        <MajorUpgrade
            DowngradeErrorMessage="A newer version of GetDist GUI is already installed."
            AllowSameVersionUpgrades="yes"
            RemoveFeatures="ALL"
            Schedule="afterInstallInitialize" />
        <MediaTemplate EmbedCab="yes" />

        <Icon Id="icon.ico" SourceFile="{icon_path}" />
        <Property Id="ARPPRODUCTICON" Value="icon.ico" />

        <!-- Properties to improve upgrade experience -->
        <Property Id="REINSTALLMODE" Value="amus" />
        <Property Id="ARPNOREPAIR" Value="yes" />
        <Property Id="ARPNOMODIFY" Value="yes" />

        <Feature Id="ProductFeature" Title="GetDist GUI" Level="1">
            <ComponentGroupRef Id="ProductComponents" />
            <ComponentRef Id="ApplicationShortcut" />
        </Feature>

        <Directory Id="TARGETDIR" Name="SourceDir">
            <Directory Id="ProgramFilesFolder">
                <Directory Id="INSTALLFOLDER" Name="GetDist GUI">
                    <!-- Components will be added by heat.exe -->
                </Directory>
            </Directory>
            <Directory Id="ProgramMenuFolder">
                <Directory Id="ApplicationProgramsFolder" Name="GetDist GUI" />
            </Directory>
        </Directory>

        <DirectoryRef Id="ApplicationProgramsFolder">
            <Component Id="ApplicationShortcut" Guid="{str(uuid.uuid4())}">
                <Shortcut Id="ApplicationStartMenuShortcut"
                          Name="GetDist GUI"
                          Description="GetDist Graphical User Interface"
                          Target="[INSTALLFOLDER]GetDistGUI.exe"
                          WorkingDirectory="INSTALLFOLDER" />
                <RemoveFolder Id="RemoveApplicationProgramsFolder" Directory="ApplicationProgramsFolder" On="uninstall" />
                <RegistryValue Root="HKCU" Key="Software\\GetDist\\GUI" Name="installed" Type="integer" Value="1" KeyPath="yes" />
            </Component>
        </DirectoryRef>
    </Product>
</Wix>
"""

    with open(wxs_path, "w", encoding="utf-8") as f:
        f.write(wxs_content)

    # Create a components file using heat.exe
    components_wxs_path = os.path.join(temp_dir, "GetDistGUIComponents.wxs")

    subprocess.check_call(
        [
            "heat",
            "dir",
            input_dir,
            "-nologo",
            "-sfrag",
            "-srd",
            "-sreg",
            "-gg",
            "-cg",
            "ProductComponents",
            "-dr",
            "INSTALLFOLDER",
            "-var",
            "var.SourceDir",
            "-out",
            components_wxs_path,
        ]
    )

    return temp_dir, wxs_path, components_wxs_path


def build_msi(temp_dir, wxs_path, components_wxs_path, input_dir, output_dir, version):
    """Build the MSI installer using WiX tools"""
    print("Building MSI installer...")

    # Compile the WiX source files
    subprocess.check_call(
        ["candle", "-nologo", "-dSourceDir=" + input_dir, wxs_path, components_wxs_path, "-out", temp_dir + "\\"]
    )

    # Link the compiled objects into an MSI
    msi_path = os.path.join(output_dir, f"GetDist-GUI-{version}.msi")

    subprocess.check_call(
        [
            "light",
            "-nologo",
            "-ext",
            "WixUIExtension",
            "-cultures:en-us",
            "-out",
            msi_path,
            os.path.join(temp_dir, "GetDistGUI.wixobj"),
            os.path.join(temp_dir, "GetDistGUIComponents.wixobj"),
        ]
    )

    print(f"MSI installer created at {msi_path}")
    return msi_path


def main():
    """Main function to parse arguments and create the MSI installer"""
    parser = argparse.ArgumentParser(description="Create an MSI installer for GetDist GUI")
    parser.add_argument(
        "--input-dir", default="dist/GetDistGUI", help="Input directory containing the PyInstaller output"
    )
    parser.add_argument("--output-dir", default="dist", help="Output directory for the MSI installer")
    parser.add_argument("--version", default=None, help="Version number for the installer")
    args = parser.parse_args()

    # Check if running on Windows
    if sys.platform != "win32":
        print("Error: This script must be run on Windows.")
        sys.exit(1)

    # Get version if not provided
    version = args.version
    if not version:
        version = find_version()

    # Check if WiX is installed
    if not check_wix_installed():
        print("Error: WiX Toolset is required to create MSI installers.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Create WiX files
        temp_dir, wxs_path, components_wxs_path = create_wix_files(args.input_dir, args.output_dir, version)

        # Build MSI
        msi_path = build_msi(temp_dir, wxs_path, components_wxs_path, args.input_dir, args.output_dir, version)

        print(f"MSI installer created successfully at {msi_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating MSI installer: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary files
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
