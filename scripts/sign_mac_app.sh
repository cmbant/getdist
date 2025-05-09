#!/bin/bash
# Script to sign and notarize a Mac app bundle
# This script handles the complex framework fixing and signing logic

set -e

# Check if an app path was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-app-bundle> [skip-notarization]"
    exit 1
fi

APP_PATH="$1"
SKIP_NOTARIZATION="${2:-false}"

# Check if the app bundle exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

# Rename app bundle to remove spaces (if needed)
if [[ "$APP_PATH" == *" "* ]]; then
    NEW_PATH=$(echo "$APP_PATH" | tr ' ' '_')
    echo "Renaming app bundle to remove spaces..."
    mv "$APP_PATH" "$NEW_PATH"
    APP_PATH="$NEW_PATH"
fi

# Create a temporary directory for the certificate
CERT_PATH=$(mktemp)
KEYCHAIN_PATH=$(mktemp).keychain-db

# Function to clean up temporary files
cleanup() {
    echo "Cleaning up temporary files..."
    rm -f "$CERT_PATH"
    security delete-keychain "$KEYCHAIN_PATH" 2>/dev/null || true
}

# Set up trap to clean up on exit
trap cleanup EXIT

# Check if MACOS_CERTIFICATE environment variable is set
if [ -z "$MACOS_CERTIFICATE" ]; then
    echo "Error: MACOS_CERTIFICATE environment variable is not set"
    exit 1
fi

# Check if MACOS_CERTIFICATE_PWD environment variable is set
if [ -z "$MACOS_CERTIFICATE_PWD" ]; then
    echo "Error: MACOS_CERTIFICATE_PWD environment variable is not set"
    exit 1
fi

# Decode and save the certificate
echo "Decoding certificate..."
echo "$MACOS_CERTIFICATE" | base64 --decode > "$CERT_PATH"

# Create a temporary keychain
echo "Creating temporary keychain..."
security create-keychain -p "temp-password" "$KEYCHAIN_PATH"
security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
security unlock-keychain -p "temp-password" "$KEYCHAIN_PATH"

# Import certificate to keychain
echo "Importing certificate to keychain..."
security import "$CERT_PATH" -P "$MACOS_CERTIFICATE_PWD" -A -t cert -f pkcs12 -k "$KEYCHAIN_PATH"
security list-keychain -d user -s "$KEYCHAIN_PATH"

# Allow codesign to access the keychain
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "temp-password" "$KEYCHAIN_PATH"

# Find the identity
echo "Available signing identities:"
IDENTITY=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed -E 's/.*"(Developer ID Application: .+)"$/\1/')
echo "Using identity: $IDENTITY"

# Create entitlements file
echo "Creating entitlements file..."
ENTITLEMENTS_PATH=$(mktemp).plist
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

# Sign the app with special handling for Qt frameworks
echo "Signing app bundle with hardened runtime..."

# Fix Qt frameworks before signing
echo "Fixing Qt frameworks..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
chmod +x "$SCRIPT_DIR/fix_qt_frameworks.sh"
"$SCRIPT_DIR/fix_qt_frameworks.sh" "$APP_PATH"

# Create a script to fix Info.plist files
FIX_PLIST_PATH=$(mktemp).sh
cat > "$FIX_PLIST_PATH" << 'EOF'
#!/bin/bash
sed -i '' '/<\/dict>/i\
<key>CFBundleSupportedPlatforms</key>\
<array>\
<string>MacOSX</string>\
</array>
' "$1"
EOF
chmod +x "$FIX_PLIST_PATH"

# First, sign all the Qt frameworks individually
echo "Signing Qt frameworks..."
find "$APP_PATH/Contents/Frameworks" -type f -name "*.dylib" -o -name "*.so" | while read -r file; do
    echo "Signing $file"
    codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$file"
done

# Sign any framework bundles - with special handling for Qt frameworks
echo "Signing framework bundles..."
find "$APP_PATH/Contents/Frameworks" -type d -name "*.framework" | while read -r framework; do
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
codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$APP_PATH/Contents/MacOS/"*

# Finally sign the app bundle
echo "Signing app bundle..."
codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$APP_PATH"

# Verify signature
echo "Verifying app signature..."
codesign --verify --verbose "$APP_PATH"

# Verify with strict validation
echo "Verifying with strict validation..."
codesign --verify --verbose=4 --strict "$APP_PATH" || {
    echo "Warning: Strict validation failed, attempting to fix frameworks..."
    
    # Check for specific issues with frameworks
    echo "Checking for framework issues..."

    # Check QtGui.framework (known problematic framework)
    if [ -d "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtGui.framework" ]; then
        echo "Verifying QtGui.framework..."
        codesign --verify --verbose "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtGui.framework" || {
            echo "Warning: QtGui.framework verification failed, attempting to fix..."
            # Try to fix the framework structure
            if [ -d "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtGui.framework/Resources" ]; then
                echo "Checking QtGui.framework Info.plist..."
                if [ -f "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtGui.framework/Resources/Info.plist" ]; then
                    echo "Adding CFBundleSupportedPlatforms to QtGui.framework Info.plist"
                    "$FIX_PLIST_PATH" "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtGui.framework/Resources/Info.plist"
                    # Re-sign the framework
                    codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtGui.framework"
                fi
            fi
        }
    fi

    # Check QtQmlMeta.framework (known problematic framework)
    if [ -d "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework" ]; then
        echo "Verifying QtQmlMeta.framework..."
        codesign --verify --verbose "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework" || {
            echo "Warning: QtQmlMeta.framework verification failed, attempting to fix..."
            # Try to fix the framework structure
            if [ -d "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework/Resources" ]; then
                echo "Checking QtQmlMeta.framework Info.plist..."
                if [ -f "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework/Resources/Info.plist" ]; then
                    echo "Adding CFBundleSupportedPlatforms to QtQmlMeta.framework Info.plist"
                    "$FIX_PLIST_PATH" "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework/Resources/Info.plist"
                    # Re-sign the framework
                    codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework"
                fi
            fi
        }
    fi

    # Check all Qt frameworks
    echo "Verifying all Qt frameworks..."
    find "$APP_PATH/Contents/Frameworks" -type d -name "Qt*.framework" | while read -r framework; do
        echo "Verifying framework: $framework"
        codesign --verify --verbose "$framework" || {
            echo "Warning: Framework verification failed for $framework, attempting to fix..."
            framework_name=$(basename "$framework" .framework)
            # Try to fix the framework structure
            if [ -d "$framework/Resources" ]; then
                echo "Checking $framework_name Info.plist..."
                if [ -f "$framework/Resources/Info.plist" ]; then
                    echo "Adding CFBundleSupportedPlatforms to $framework_name Info.plist"
                    "$FIX_PLIST_PATH" "$framework/Resources/Info.plist"
                    # Re-sign the framework
                    codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$framework"
                fi
            fi
        }
    done
    
    # Re-sign the app bundle after fixing frameworks
    echo "Re-signing app bundle after framework fixes..."
    codesign --force --verify --verbose --options runtime --entitlements "$ENTITLEMENTS_PATH" --sign "$IDENTITY" "$APP_PATH"
}

# Skip notarization if requested
if [ "$SKIP_NOTARIZATION" = "true" ]; then
    echo "Skipping notarization as requested"
    exit 0
fi
