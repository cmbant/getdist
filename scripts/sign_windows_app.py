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
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

KMS_CNG_PROVIDER_RELEASES_API = "https://api.github.com/repos/GoogleCloudPlatform/kms-integrations/releases?per_page=20"
KMS_CNG_PROVIDER_URL_ENV_VAR = "GCP_KMS_CNG_PROVIDER_URL"
KMS_CNG_RELEASE_TAG_PREFIX = "cng-v"
KMS_CNG_ARCHIVE_NAME_RE = re.compile(r"^kmscng-[0-9][0-9A-Za-z.\-]*-windows-amd64\.zip$")


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


def download_url_to_file(url, destination):
    """Download a URL to a local file using a GitHub-friendly user-agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "getdist-sign-windows/1.0"})
    with urllib.request.urlopen(request) as response, open(destination, "wb") as file_handle:
        file_handle.write(response.read())


def fetch_kms_integrations_releases():
    """Fetch the release list for GoogleCloudPlatform/kms-integrations."""
    request = urllib.request.Request(
        KMS_CNG_PROVIDER_RELEASES_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "getdist-sign-windows/1.0",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def get_cng_release_version(tag_name):
    """Convert a CNG release tag like cng-v1.3 into a sortable version tuple."""
    if not tag_name or not tag_name.startswith(KMS_CNG_RELEASE_TAG_PREFIX):
        return ()

    version = tag_name[len(KMS_CNG_RELEASE_TAG_PREFIX) :]
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def is_supported_cng_release(release):
    """Return True for non-draft, non-prerelease CNG releases."""
    tag_name = release.get("tag_name", "")
    return (
        tag_name.startswith(KMS_CNG_RELEASE_TAG_PREFIX) and not release.get("draft") and not release.get("prerelease")
    )


def is_cng_windows_archive(asset_name):
    """Return True when the asset name matches the published Windows CNG zip."""
    return bool(KMS_CNG_ARCHIVE_NAME_RE.fullmatch(asset_name or ""))


def resolve_kms_cng_provider_asset(releases):
    """Resolve the newest Windows CNG provider asset from the GitHub releases list."""
    candidates = sorted(
        (release for release in releases if is_supported_cng_release(release)),
        key=lambda release: (
            get_cng_release_version(release.get("tag_name")),
            release.get("published_at") or "",
        ),
        reverse=True,
    )

    for release in candidates:
        for asset in release.get("assets", []):
            if is_cng_windows_archive(asset.get("name", "")):
                return release, asset

    raise RuntimeError(
        "Could not find a published Windows Cloud KMS CNG provider asset in "
        "GoogleCloudPlatform/kms-integrations releases."
    )


def resolve_kms_cng_provider_download():
    """Resolve the CNG provider download URL and local filename."""
    override_url = os.environ.get(KMS_CNG_PROVIDER_URL_ENV_VAR)
    if override_url:
        asset_name = os.path.basename(urllib.parse.urlparse(override_url).path) or "kmscng-provider-download"
        print(f"Using {KMS_CNG_PROVIDER_URL_ENV_VAR} override: {override_url}")
        return override_url, asset_name

    release, asset = resolve_kms_cng_provider_asset(fetch_kms_integrations_releases())
    print(f"Resolved Google Cloud KMS CNG provider release {release['tag_name']} ({asset['name']})")
    return asset["browser_download_url"], asset["name"]


def find_installer_msi(search_root):
    """Find the extracted MSI installer path inside a directory tree."""
    matches = []
    for root, _, files in os.walk(search_root):
        for filename in files:
            if filename.lower().endswith(".msi"):
                matches.append(os.path.join(root, filename))

    if not matches:
        raise RuntimeError(f"Could not find an MSI installer under: {search_root}")

    return sorted(matches)[0]


def install_kms_cng_provider():
    """Download and install the Google Cloud KMS CNG provider."""
    tools_dir = os.path.join(tempfile.gettempdir(), "signing-tools")
    os.makedirs(tools_dir, exist_ok=True)

    download_url, asset_name = resolve_kms_cng_provider_download()
    package_path = os.path.join(tools_dir, asset_name)

    if not os.path.exists(package_path):
        print("Downloading Google Cloud KMS CNG provider...")
        download_url_to_file(download_url, package_path)
        print(f"Downloaded to: {package_path}")

    installer_path = package_path
    if package_path.lower().endswith(".zip"):
        extract_dir = os.path.join(tools_dir, f"{os.path.splitext(asset_name)[0]}-extracted")
        if not os.path.isdir(extract_dir):
            print("Extracting Google Cloud KMS CNG provider archive...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(package_path) as archive:
                archive.extractall(extract_dir)
        installer_path = find_installer_msi(extract_dir)
    elif not package_path.lower().endswith(".msi"):
        raise RuntimeError(f"Unsupported Cloud KMS CNG provider package: {package_path}")

    print("Installing Google Cloud KMS CNG provider...")
    subprocess.check_call(
        ["msiexec", "/i", installer_path, "/quiet", "/norestart"],
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
