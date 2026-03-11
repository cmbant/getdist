#!/usr/bin/env python
"""
Script to sign Windows executables and DLLs using signtool with Google Cloud KMS.

Uses the Google Cloud KMS CNG provider with signtool.exe, matching the same
approach used for local signing.

Required environment variables:
    GCP_KMS_KEY:           Full KMS key resource path, e.g.
                           projects/PROJECT/locations/LOCATION/keyRings/KEYRING/
                           cryptoKeys/KEY/cryptoKeyVersions/VERSION
    GCP_CERTIFICATE_CHAIN: Base64-encoded certificate chain file (.crt)

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

KMS_CNG_PROVIDER_URL = "https://storage.googleapis.com/cloud-kms-cng-provider/cloud-kms-cng-provider-setup.msi"


def find_signtool():
    """Find signtool.exe on the system."""
    # Check PATH first
    try:
        subprocess.check_call(["where", "signtool"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "signtool"
    except subprocess.CalledProcessError:
        pass

    # Scan Windows Kits for the latest version
    windows_kits_dir = r"C:\Program Files (x86)\Windows Kits\10\bin"
    if os.path.exists(windows_kits_dir):
        versions = sorted(
            (
                d
                for d in os.listdir(windows_kits_dir)
                if os.path.isdir(os.path.join(windows_kits_dir, d)) and d.startswith("10.")
            ),
            reverse=True,
        )
        for version in versions:
            signtool_path = os.path.join(windows_kits_dir, version, "x64", "signtool.exe")
            if os.path.exists(signtool_path):
                print(f"Found signtool at: {signtool_path}")
                return signtool_path

    print("Error: Could not find signtool.exe. Install the Windows SDK or add signtool to your PATH.")
    return None


def install_kms_cng_provider():
    """Download and install the Google Cloud KMS CNG provider."""
    import urllib.request

    tools_dir = os.path.join(tempfile.gettempdir(), "signing-tools")
    os.makedirs(tools_dir, exist_ok=True)
    msi_path = os.path.join(tools_dir, "cloud-kms-cng-provider-setup.msi")

    if not os.path.exists(msi_path):
        print("Downloading Google Cloud KMS CNG provider...")
        urllib.request.urlretrieve(KMS_CNG_PROVIDER_URL, msi_path)
        print(f"Downloaded to: {msi_path}")

    print("Installing Google Cloud KMS CNG provider...")
    subprocess.check_call(
        ["msiexec", "/i", msi_path, "/quiet", "/norestart"],
    )
    print("CNG provider installed successfully")


def collect_files_to_sign(path):
    """Collect all signable files from a path (file or directory)."""
    signable_extensions = ("*.exe", "*.dll", "*.pyd", "*.msi")

    if os.path.isfile(path):
        print(f"Signing single file: {path}")
        return [path]

    print(f"Signing files in directory: {path}")
    files_to_sign = set()
    for root, _, _ in os.walk(path):
        for ext in signable_extensions:
            files_to_sign.update(glob.glob(os.path.join(root, ext)))

    return sorted(files_to_sign)


def sign_files(path, signtool_path, kms_key, certfile):
    """Sign all executable files using signtool with Google Cloud KMS CNG provider."""
    files_to_sign = collect_files_to_sign(path)
    print(f"Found {len(files_to_sign)} files to sign")

    if not files_to_sign:
        print("No files to sign")
        return True

    failed = []
    for file_path in files_to_sign:
        print(f"Signing {file_path}...")
        try:
            subprocess.check_call(
                [
                    signtool_path,
                    "sign",
                    "/v",
                    "/fd",
                    "sha256",
                    "/td",
                    "sha256",
                    "/tr",
                    "http://timestamp.digicert.com",
                    "/f",
                    certfile,
                    "/csp",
                    "Google Cloud KMS Provider",
                    "/kc",
                    kms_key,
                    "/d",
                    "GetDist GUI",
                    file_path,
                ]
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to sign {file_path}: {e}")
            failed.append(file_path)

    if failed:
        print(f"Warning: {len(failed)} file(s) failed to sign")
    else:
        print("All files signed successfully")
    return len(failed) == 0


def main():
    """Main function to parse arguments and sign files."""
    parser = argparse.ArgumentParser(description="Sign Windows executables and DLLs using Google Cloud KMS")
    parser.add_argument("--dir", required=True, help="Directory or file to sign")
    parser.add_argument(
        "--install-cng",
        action="store_true",
        help="Install the Google Cloud KMS CNG provider (for CI)",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        print("Error: This script must be run on Windows.")
        sys.exit(1)

    # Read configuration from environment
    kms_key = os.environ.get("GCP_KMS_KEY")
    cert_chain_b64 = os.environ.get("GCP_CERTIFICATE_CHAIN")

    missing = []
    if not kms_key:
        missing.append("GCP_KMS_KEY")
    if not cert_chain_b64:
        missing.append("GCP_CERTIFICATE_CHAIN")
    if missing:
        print(f"Error: Missing required environment variable(s): {', '.join(missing)}")
        sys.exit(1)

    # Optionally install the CNG provider (needed on CI runners)
    if args.install_cng:
        install_kms_cng_provider()

    # Find signtool
    signtool_path = find_signtool()
    if not signtool_path:
        sys.exit(1)

    # Write certificate chain to temp file
    tools_dir = os.path.join(tempfile.gettempdir(), "signing-tools")
    os.makedirs(tools_dir, exist_ok=True)
    cert_file = os.path.join(tools_dir, "certificate-chain.crt")
    cert_chain_data = base64.b64decode(cert_chain_b64)
    with open(cert_file, "wb") as f:
        f.write(cert_chain_data)

    try:
        success = sign_files(args.dir, signtool_path, kms_key, cert_file)
        if not success:
            sys.exit(1)
    finally:
        if os.path.exists(cert_file):
            os.unlink(cert_file)


if __name__ == "__main__":
    main()
