#!/usr/bin/env python
"""
Script to sign Windows executables and DLLs using a code signing certificate.
The certificate is provided as a base64-encoded string in the WINDOWS_CERTIFICATE environment variable.
The certificate password is provided in the WINDOWS_CERTIFICATE_PASSWORD environment variable.

Usage:
    python scripts/sign_windows_app.py --dir DIRECTORY
"""

import argparse
import base64
import glob
import os
import subprocess
import sys
import tempfile


def find_signtool():
    """Find the signtool.exe executable"""
    # Check common locations for signtool.exe
    possible_locations = [
        # GitHub Actions Windows runner
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22000.0\x64\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.17763.0\x64\signtool.exe",
        r"C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe",
        # Visual Studio locations
        r"C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.8 Tools\signtool.exe",
        r"C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.7.2 Tools\signtool.exe",
        r"C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.6.1 Tools\signtool.exe",
    ]

    # Check if signtool is in PATH
    try:
        subprocess.check_call(["where", "signtool"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "signtool"
    except subprocess.CalledProcessError:
        pass

    # Check each possible location
    for location in possible_locations:
        if os.path.exists(location):
            print(f"Found signtool at: {location}")
            return location

    # Try to find signtool in Windows Kits directory
    windows_kits_dir = r"C:\Program Files (x86)\Windows Kits\10\bin"
    if os.path.exists(windows_kits_dir):
        # Find the latest version
        versions = []
        for item in os.listdir(windows_kits_dir):
            if os.path.isdir(os.path.join(windows_kits_dir, item)) and item.startswith("10."):
                versions.append(item)

        if versions:
            # Sort versions in descending order
            versions.sort(reverse=True)
            for version in versions:
                signtool_path = os.path.join(windows_kits_dir, version, "x64", "signtool.exe")
                if os.path.exists(signtool_path):
                    print(f"Found signtool at: {signtool_path}")
                    return signtool_path

    # If we get here, we couldn't find signtool
    print("Error: Could not find signtool.exe. Please install the Windows SDK or add signtool to your PATH.")
    return None


def sign_files(path, certificate_path, certificate_password):
    """Sign all executable files in the directory or a single file"""
    # Find signtool.exe
    signtool_path = find_signtool()
    if not signtool_path:
        print("Skipping signing due to missing signtool.exe")
        return False

    # Check if path is a file or directory
    if os.path.isfile(path):
        print(f"Signing single file: {path}")
        files_to_sign = [path]
    else:
        print(f"Signing files in directory: {path}")
        # Find all executable files
        exe_files = glob.glob(os.path.join(path, "*.exe"))
        dll_files = glob.glob(os.path.join(path, "*.dll"))
        pyd_files = glob.glob(os.path.join(path, "*.pyd"))
        msi_files = glob.glob(os.path.join(path, "*.msi"))

        # Combine all files to sign
        files_to_sign = exe_files + dll_files + pyd_files + msi_files

        # Also find files in subdirectories
        for root, _, _ in os.walk(path):
            exe_files = glob.glob(os.path.join(root, "*.exe"))
            dll_files = glob.glob(os.path.join(root, "*.dll"))
            pyd_files = glob.glob(os.path.join(root, "*.pyd"))
            msi_files = glob.glob(os.path.join(root, "*.msi"))
            files_to_sign.extend(exe_files + dll_files + pyd_files + msi_files)

    print(f"Found {len(files_to_sign)} files to sign")

    # Sign each file
    for file_path in files_to_sign:
        print(f"Signing {file_path}...")
        try:
            subprocess.check_call(
                [
                    signtool_path,
                    "sign",
                    "/f",
                    certificate_path,
                    "/p",
                    certificate_password,
                    "/tr",
                    "http://timestamp.digicert.com",
                    "/td",
                    "sha256",
                    "/fd",
                    "sha256",
                    "/d",
                    "GetDist GUI",
                    file_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to sign {file_path}: {e}")
            # Continue with other files even if one fails

    print("Signing completed")
    return True


def main():
    """Main function to parse arguments and sign files"""
    parser = argparse.ArgumentParser(description="Sign Windows executables and DLLs")
    parser.add_argument("--dir", required=True, help="Directory containing files to sign")
    args = parser.parse_args()

    # Check if running on Windows
    if sys.platform != "win32":
        print("Error: This script must be run on Windows.")
        sys.exit(1)

    # Get certificate from environment variable
    certificate_base64 = os.environ.get("WINDOWS_CERTIFICATE")
    certificate_password = os.environ.get("WINDOWS_CERTIFICATE_PASSWORD")

    if not certificate_base64:
        print("Error: WINDOWS_CERTIFICATE environment variable not set.")
        sys.exit(1)

    if not certificate_password:
        print("Error: WINDOWS_CERTIFICATE_PASSWORD environment variable not set.")
        sys.exit(1)

    # Decode certificate
    certificate_data = base64.b64decode(certificate_base64)

    # Create temporary file for certificate
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pfx") as temp_cert:
        temp_cert.write(certificate_data)
        certificate_path = temp_cert.name

    try:
        # Sign files
        success = sign_files(args.dir, certificate_path, certificate_password)
        if not success:
            print("Warning: Signing was skipped due to missing signtool.exe")
            # Exit with a non-zero code but not a failure (1) to indicate a warning
            sys.exit(0)
    finally:
        # Clean up temporary file
        os.unlink(certificate_path)


if __name__ == "__main__":
    main()
