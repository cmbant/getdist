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
5. Optionally, check "Sign and notarize the app" if you have set up the required secrets

The workflow will:
- Build the Mac app bundle
- Create a DMG installer
- Sign and notarize the app (if enabled and secrets are configured)
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
2. Sign the app using:

   ```bash
   codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name (TEAM_ID)" "dist/GetDist GUI.app"
   ```

3. Notarize the app for distribution (requires Apple Developer account)

For detailed instructions on setting up code signing and notarization in GitHub Actions, see [MAC_APP_SIGNING.md](MAC_APP_SIGNING.md).

## Notes

- The built app includes all necessary Python dependencies and does not require a separate Python installation
- The app should work on macOS 10.7 or later
- The build process uses a dedicated project directory to avoid conflicts with your system Python
