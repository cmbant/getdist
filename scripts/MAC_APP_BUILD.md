# Building GetDist GUI Mac App

This document explains how to build a standalone Mac application bundle for the GetDist GUI.

## Prerequisites

- macOS operating system
- Python 3.9 or later

## Building Locally

The simplest way to build the app is to use the provided shell script:

```bash
# Basic build
./scripts/build_mac_app.sh

# Customize output and project directories
./scripts/build_mac_app.sh --output-dir custom_dist --project-dir custom_env
```

The script will:
1. Create a dedicated virtual environment in the specified project directory
2. Install all required dependencies
3. Build a Mac app bundle using PyInstaller
4. Create a DMG installer for easy distribution

## Building with GitHub Actions

The repository includes a GitHub workflow that automatically builds the Mac app when changes are pushed to the main branch or when manually triggered.

To manually trigger a build:

1. Go to the GitHub repository
2. Click on the "Actions" tab
3. Select the "Build Mac App" workflow
4. Click "Run workflow"
5. By default, the app will be signed and notarized if the required secrets are set up
6. If you want to skip signing (for testing purposes), check the "Skip signing and notarization" option

The workflow will:
- Build the Mac app bundle
- Create a DMG installer
- Sign and notarize the app (unless explicitly skipped)
- Upload the DMG as an artifact that can be downloaded from the workflow run page

## Architecture Support

- The app is built for the architecture of the Mac it's running on (Intel or ARM)
- Apps built on Intel Macs can run on Apple Silicon Macs via Rosetta 2
- Apps built on Apple Silicon Macs will only run on Apple Silicon Macs

## Customizing the Build

You can customize the build by modifying the `scripts/build_mac_app.py` script:

- Change the output directory with the `--output-dir` parameter
- Change the project directory with the `--project-dir` parameter
- Modify the PyInstaller spec file to include additional files or dependencies

## Code Signing and Notarization

For distribution outside of development, the app should be signed and notarized:

1. Obtain an Apple Developer ID
2. Set up the required secrets for signing and notarization
3. The GitHub workflow will automatically handle the signing and notarization process

The workflow uses the [toitlang/action-macos-sign-notarize](https://github.com/toitlang/action-macos-sign-notarize) GitHub Action to handle the signing and notarization process, which greatly simplifies the workflow and makes it more reliable.

For detailed instructions on setting up code signing and notarization in GitHub Actions, see [MAC_APP_SIGNING.md](MAC_APP_SIGNING.md).

## Notes

- The built app includes all necessary Python dependencies and does not require a separate Python installation
- The app should work on macOS 10.7 or later
- The build process uses a dedicated project directory to avoid conflicts with your system Python
