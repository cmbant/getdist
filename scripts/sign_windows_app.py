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
import os
import subprocess
import sys
import tempfile
import glob


def sign_files(directory, certificate_path, certificate_password):
    """Sign all executable files in the directory"""
    print(f"Signing files in {directory}...")
    
    # Find all executable files
    exe_files = glob.glob(os.path.join(directory, "*.exe"))
    dll_files = glob.glob(os.path.join(directory, "*.dll"))
    pyd_files = glob.glob(os.path.join(directory, "*.pyd"))
    
    # Combine all files to sign
    files_to_sign = exe_files + dll_files + pyd_files
    
    # Also find files in subdirectories
    for root, _, _ in os.walk(directory):
        exe_files = glob.glob(os.path.join(root, "*.exe"))
        dll_files = glob.glob(os.path.join(root, "*.dll"))
        pyd_files = glob.glob(os.path.join(root, "*.pyd"))
        files_to_sign.extend(exe_files + dll_files + pyd_files)
    
    print(f"Found {len(files_to_sign)} files to sign")
    
    # Sign each file
    for file_path in files_to_sign:
        print(f"Signing {file_path}...")
        try:
            subprocess.check_call([
                "signtool", "sign",
                "/f", certificate_path,
                "/p", certificate_password,
                "/tr", "http://timestamp.digicert.com",
                "/td", "sha256",
                "/fd", "sha256",
                "/d", "GetDist GUI",
                file_path
            ])
        except subprocess.CalledProcessError as e:
            print(f"Failed to sign {file_path}: {e}")
            # Continue with other files even if one fails
    
    print("Signing completed")


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
        sign_files(args.dir, certificate_path, certificate_password)
    finally:
        # Clean up temporary file
        os.unlink(certificate_path)


if __name__ == "__main__":
    main()
