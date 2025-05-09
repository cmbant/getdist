#!/bin/bash
# Script to fix the "bundle format is ambiguous" error in Qt frameworks
# This script specifically targets the QtGui.framework and QtQmlMeta.framework

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

echo "Fixing bundle ambiguity issues in $APP_PATH..."

# Function to fix a specific framework
fix_framework() {
    local framework_path="$1"
    local framework_name=$(basename "$framework_path" .framework)

    echo "Fixing bundle ambiguity for: $framework_name"

    # Check if the framework exists
    if [ ! -d "$framework_path" ]; then
        echo "  Framework not found, skipping"
        return
    fi

    # Create a proper framework structure
    echo "  Creating proper framework structure..."

    # 1. Create Versions directory if it doesn't exist
    if [ ! -d "$framework_path/Versions" ]; then
        mkdir -p "$framework_path/Versions/A"
    fi

    # 2. Create Current symlink if it doesn't exist
    if [ ! -L "$framework_path/Versions/Current" ]; then
        # Find the first version directory
        local version_dir=$(find "$framework_path/Versions" -mindepth 1 -maxdepth 1 -type d | head -1)
        if [ -z "$version_dir" ]; then
            # If no version directory exists, use A
            version_dir="$framework_path/Versions/A"
            mkdir -p "$version_dir"
        fi
        local version_name=$(basename "$version_dir")
        echo "  Creating Current symlink to $version_name"
        ln -sf "$version_name" "$framework_path/Versions/Current"
    fi

    # 3. Create Resources directory if it doesn't exist
    if [ ! -d "$framework_path/Versions/Current/Resources" ]; then
        mkdir -p "$framework_path/Versions/Current/Resources"
    fi

    # 4. Create Resources symlink if it doesn't exist
    if [ ! -L "$framework_path/Resources" ]; then
        echo "  Creating Resources symlink"
        ln -sf "Versions/Current/Resources" "$framework_path/Resources"
    fi

    # 5. Create Info.plist with proper structure
    local info_plist="$framework_path/Resources/Info.plist"
    echo "  Creating Info.plist at $info_plist"
    cat > "$info_plist" << EOF
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
    <key>CFBundleName</key>
    <string>${framework_name}</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>6.0</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
    <key>CFBundleVersion</key>
    <string>6.0</string>
    <key>CSResourcesFileMapped</key>
    <true/>
    <key>MinimumOSVersion</key>
    <string>10.13</string>
</dict>
</plist>
EOF

    # 6. Move the framework binary to the Versions/Current directory if needed
    if [ -f "$framework_path/$framework_name" ] && [ ! -f "$framework_path/Versions/Current/$framework_name" ]; then
        echo "  Moving framework binary to Versions/Current"
        # Use ditto instead of cp to preserve file attributes and metadata
        ditto "$framework_path/$framework_name" "$framework_path/Versions/Current/$framework_name"
        rm "$framework_path/$framework_name"
    fi

    # 7. Create framework binary symlink if it doesn't exist
    if [ ! -L "$framework_path/$framework_name" ] && [ -f "$framework_path/Versions/Current/$framework_name" ]; then
        echo "  Creating framework binary symlink"
        ln -sf "Versions/Current/$framework_name" "$framework_path/$framework_name"
    elif [ ! -f "$framework_path/Versions/Current/$framework_name" ] && [ -f "$framework_path/$framework_name" ]; then
        echo "  Moving framework binary to Versions/Current"
        mkdir -p "$framework_path/Versions/Current"
        # Use ditto instead of cp to preserve file attributes and metadata
        ditto "$framework_path/$framework_name" "$framework_path/Versions/Current/$framework_name"
        rm "$framework_path/$framework_name"
        ln -sf "Versions/Current/$framework_name" "$framework_path/$framework_name"
    fi

    echo "  Framework structure fixed"
}

# Fix QtGui.framework
QTGUI_FRAMEWORK="$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtGui.framework"
if [ -d "$QTGUI_FRAMEWORK" ]; then
    fix_framework "$QTGUI_FRAMEWORK"
else
    echo "QtGui.framework not found at $QTGUI_FRAMEWORK"
fi

# Fix QtQmlMeta.framework
QTQMLMETA_FRAMEWORK="$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtQmlMeta.framework"
if [ -d "$QTQMLMETA_FRAMEWORK" ]; then
    fix_framework "$QTQMLMETA_FRAMEWORK"
else
    echo "QtQmlMeta.framework not found at $QTQMLMETA_FRAMEWORK"
fi

# Find and fix all Qt frameworks
echo "Searching for all Qt frameworks..."
QT_FRAMEWORKS=$(find "$APP_PATH/Contents/Frameworks" -type d -name "Qt*.framework")
for framework in $QT_FRAMEWORKS; do
    fix_framework "$framework"
done

echo "Bundle ambiguity issues fixed!"
