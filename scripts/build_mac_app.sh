#!/bin/bash
# Simple script to build the GetDist GUI Mac app
# This script should be run on a Mac

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script must be run on macOS"
    exit 1
fi

# Get the script directory and repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set up variables
OUTPUT_DIR="$REPO_ROOT/dist"
PROJECT_DIR="$REPO_ROOT/build_env"
APP_NAME="GetDist GUI.app"
DMG_NAME="GetDist-GUI.dmg"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --project-dir) PROJECT_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the app
echo "Building Mac app..."
cd "$REPO_ROOT"
python "$SCRIPT_DIR/build_mac_app.py" --output-dir "$OUTPUT_DIR" --project-dir "$PROJECT_DIR"

# Create DMG if the app was built successfully
if [ -d "$OUTPUT_DIR/$APP_NAME" ]; then
    echo "Creating DMG..."
    
    # Create a temporary directory for the DMG contents
    mkdir -p "$OUTPUT_DIR/dmg"
    
    # Copy the app bundle to the temporary directory
    cp -r "$OUTPUT_DIR/$APP_NAME" "$OUTPUT_DIR/dmg/"
    
    # Create a symlink to the Applications folder
    ln -s /Applications "$OUTPUT_DIR/dmg/"
    
    # Create the DMG
    hdiutil create -volname "GetDist GUI" -srcfolder "$OUTPUT_DIR/dmg" -ov -format UDZO "$OUTPUT_DIR/$DMG_NAME"
    
    # Clean up
    rm -rf "$OUTPUT_DIR/dmg"
    
    echo "DMG created at $OUTPUT_DIR/$DMG_NAME"
else
    echo "Error: App build failed, skipping DMG creation"
    exit 1
fi

echo "Build process completed successfully!"
echo "App bundle: $OUTPUT_DIR/$APP_NAME"
echo "DMG installer: $OUTPUT_DIR/$DMG_NAME"
