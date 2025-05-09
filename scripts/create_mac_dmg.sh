#!/bin/bash
# Script to create a DMG for a Mac app bundle

set -e

# Check if an app path was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-app-bundle> [output-dmg-path]"
    exit 1
fi

APP_PATH="$1"
DMG_PATH="${2:-dist/GetDist-GUI.dmg}"

# Check if the app bundle exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
DMG_DIR=$(dirname "$DMG_PATH")
mkdir -p "$DMG_DIR"

# Create a DMG with the app
echo "Creating DMG with app..."

# Create a temporary directory for the DMG contents
TEMP_DIR=$(mktemp -d)

# Copy the app bundle to the temporary directory
# Use the original name with spaces for the app in the DMG
APP_NAME=$(basename "$APP_PATH" | sed 's/_/ /g')
# Use ditto instead of cp -r to preserve symbolic links and file attributes
echo "Copying app bundle with ditto to preserve symbolic links..."
ditto "$APP_PATH" "$TEMP_DIR/$APP_NAME"

# Create a symlink to the Applications folder
ln -s /Applications "$TEMP_DIR/"

# Create the DMG
hdiutil create -volname "GetDist GUI" -srcfolder "$TEMP_DIR" -ov -format UDZO "$DMG_PATH"

# Clean up
rm -rf "$TEMP_DIR"

echo "DMG created at $DMG_PATH"

# Verify the app signature and notarization
echo "Verifying app signature..."
codesign --verify --verbose "$APP_PATH"
echo "Verifying app notarization..."
spctl --assess --verbose=4 "$APP_PATH"

echo "DMG creation process completed"
