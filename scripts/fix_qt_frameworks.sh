#!/bin/bash
# Script to fix Qt frameworks in a macOS app bundle
# This script addresses the "bundle format is ambiguous" error

set -e

# Check if an app path was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path-to-app-bundle>"
    exit 1
fi

APP_PATH="$1"

# Check if the app bundle exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

echo "Fixing Qt frameworks in $APP_PATH..."

# Function to fix a framework
fix_framework() {
    local framework_path="$1"
    local framework_name=$(basename "$framework_path" .framework)
    
    echo "Processing framework: $framework_name"
    
    # Check if the framework exists
    if [ ! -d "$framework_path" ]; then
        echo "  Framework not found, skipping"
        return
    fi
    
    # Check if the framework has an Info.plist
    if [ ! -f "$framework_path/Resources/Info.plist" ] && [ ! -f "$framework_path/Versions/Current/Resources/Info.plist" ]; then
        echo "  Creating Info.plist for $framework_name"
        
        # Create Resources directory if it doesn't exist
        if [ -d "$framework_path/Versions/Current" ]; then
            mkdir -p "$framework_path/Versions/Current/Resources"
            INFO_PLIST_PATH="$framework_path/Versions/Current/Resources/Info.plist"
        else
            mkdir -p "$framework_path/Resources"
            INFO_PLIST_PATH="$framework_path/Resources/Info.plist"
        fi
        
        # Create a basic Info.plist
        cat > "$INFO_PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>${framework_name}</string>
    <key>CFBundleIdentifier</key>
    <string>org.qt-project.${framework_name}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
</dict>
</plist>
EOF
    else
        echo "  Info.plist already exists for $framework_name"
    fi
}

# Find all Qt frameworks in the app bundle
echo "Searching for Qt frameworks..."
QT_FRAMEWORKS=$(find "$APP_PATH/Contents/Frameworks" -type d -name "*.framework" | grep -i "Qt")

# Process each framework
for framework in $QT_FRAMEWORKS; do
    fix_framework "$framework"
done

echo "Fixing framework symlinks..."
# Fix symlinks in the Versions directory
for framework in $QT_FRAMEWORKS; do
    if [ -d "$framework/Versions" ]; then
        echo "Checking symlinks in $framework"
        
        # Check if Current symlink exists
        if [ ! -L "$framework/Versions/Current" ] && [ -d "$framework/Versions/A" ]; then
            echo "  Creating Current symlink"
            ln -sf A "$framework/Versions/Current"
        fi
        
        # Get the framework name
        framework_name=$(basename "$framework" .framework)
        
        # Check if the framework binary symlink exists
        if [ ! -L "$framework/$framework_name" ] && [ -f "$framework/Versions/Current/$framework_name" ]; then
            echo "  Creating framework binary symlink"
            ln -sf "Versions/Current/$framework_name" "$framework/$framework_name"
        fi
        
        # Check if the Resources symlink exists
        if [ ! -L "$framework/Resources" ] && [ -d "$framework/Versions/Current/Resources" ]; then
            echo "  Creating Resources symlink"
            ln -sf "Versions/Current/Resources" "$framework/Resources"
        fi
    fi
done

echo "Qt frameworks fixed successfully!"
