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

# Detect architecture
if [[ $(uname -m) == "arm64" ]]; then
    ARCH="arm"
else
    ARCH="intel"
fi
DMG_NAME="GetDist-GUI-$ARCH.dmg"

echo "Detected architecture: $ARCH"

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

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify uv is working
echo "Verifying uv installation..."
uv --version

# Clean up any existing environment
if [ -d "$PROJECT_DIR" ]; then
    echo "Removing existing environment at $PROJECT_DIR"
    rm -rf "$PROJECT_DIR"
fi

# Build the app
echo "Building Mac app..."
cd "$REPO_ROOT"
python "$SCRIPT_DIR/build_mac_app.py" --output-dir "$OUTPUT_DIR" --project-dir "$PROJECT_DIR"

# Create DMG if the app was built successfully
if [ -d "$OUTPUT_DIR/$APP_NAME" ]; then
    # Create entitlements file for hardened runtime if it doesn't exist
    ENTITLEMENTS_PATH="$REPO_ROOT/entitlements.plist"
    if [ ! -f "$ENTITLEMENTS_PATH" ]; then
        echo "Creating entitlements file..."
        cat > "$ENTITLEMENTS_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
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
EOF
    fi

    # Check if we have a Developer ID certificate for signing
    if command -v codesign &> /dev/null && codesign -v &> /dev/null; then
        IDENTITY=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed -E 's/.*"(Developer ID Application: .+)"$/\1/')
        if [ ! -z "$IDENTITY" ]; then
            echo "Signing app with identity: $IDENTITY"

            # Fix Qt frameworks before signing
            echo "Fixing Qt frameworks..."
            chmod +x "$SCRIPT_DIR/fix_qt_frameworks.sh"
            "$SCRIPT_DIR/fix_qt_frameworks.sh" "$OUTPUT_DIR/$APP_NAME"

            # First, sign all the Qt frameworks individually
            echo "Signing Qt frameworks..."
            find "$OUTPUT_DIR/$APP_NAME/Contents/Frameworks" -type f -name "*.dylib" -o -name "*.so" | while read -r file; do
                echo "Signing $file"
                codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$file"
            done

            # Sign any framework bundles - with special handling for Qt frameworks
            echo "Signing framework bundles..."
            find "$OUTPUT_DIR/$APP_NAME/Contents/Frameworks" -type d -name "*.framework" | while read -r framework; do
                echo "Signing framework: $framework"

                # Sign the framework binary first if it exists
                framework_name=$(basename "$framework" .framework)
                if [ -f "$framework/$framework_name" ]; then
                    echo "  Signing framework binary: $framework/$framework_name"
                    codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$framework/$framework_name"
                fi

                # Sign the framework bundle
                codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$framework"
            done

            # Sign the main executable
            echo "Signing main executable..."
            codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$OUTPUT_DIR/$APP_NAME/Contents/MacOS/GetDistGUI"

            # Finally sign the app bundle
            echo "Signing app bundle..."
            codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$OUTPUT_DIR/$APP_NAME"

            echo "Verifying signature..."
            codesign --verify --verbose "$OUTPUT_DIR/$APP_NAME"

            # Verify with strict validation
            echo "Verifying with strict validation..."
            codesign --verify --verbose=4 --strict "$OUTPUT_DIR/$APP_NAME"

            # Check for specific issues with frameworks
            echo "Checking for framework issues..."

            # Check QtQuick.framework
            if [ -d "$OUTPUT_DIR/$APP_NAME/Contents/Frameworks/PySide6/Qt/lib/QtQuick.framework" ]; then
                echo "Verifying QtQuick.framework..."
                codesign --verify --verbose "$OUTPUT_DIR/$APP_NAME/Contents/Frameworks/PySide6/Qt/lib/QtQuick.framework"
            fi

            # Check QtQmlMeta.framework
            if [ -d "$OUTPUT_DIR/$APP_NAME/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework" ]; then
                echo "Verifying QtQmlMeta.framework..."
                codesign --verify --verbose "$OUTPUT_DIR/$APP_NAME/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework"
            fi

            # Check all Qt frameworks
            echo "Verifying all Qt frameworks..."
            find "$OUTPUT_DIR/$APP_NAME/Contents/Frameworks" -type d -name "Qt*.framework" | while read -r framework; do
                echo "Verifying framework: $framework"
                codesign --verify --verbose "$framework"
            done
        else
            echo "No Developer ID certificate found, skipping signing"
        fi
    else
        echo "codesign tool not available, skipping signing"
    fi

    echo "Creating DMG..."

    # Create a temporary directory for the DMG contents
    mkdir -p "$OUTPUT_DIR/dmg"

    # Copy the app bundle to the temporary directory
    cp -r "$OUTPUT_DIR/$APP_NAME" "$OUTPUT_DIR/dmg/"

    # Create a symlink to the Applications folder
    ln -s /Applications "$OUTPUT_DIR/dmg/"

    # Create the DMG
    hdiutil create -volname "GetDist GUI" -srcfolder "$OUTPUT_DIR/dmg" -ov -format UDZO "$OUTPUT_DIR/$DMG_NAME"

    # Sign the DMG if we have a Developer ID certificate
    if [ ! -z "$IDENTITY" ]; then
        echo "Signing DMG..."
        codesign --sign "$IDENTITY" "$OUTPUT_DIR/$DMG_NAME"
    fi

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
