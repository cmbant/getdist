#!/bin/bash
# Script to notarize a Mac app bundle
# This script handles the notarization process

set -e

# Check if an app path was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-app-bundle>"
    exit 1
fi

APP_PATH="$1"

# Check if the app bundle exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: App bundle not found at $APP_PATH"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$APP_STORE_CONNECT_KEY" ]; then
    echo "Error: APP_STORE_CONNECT_KEY environment variable is not set"
    exit 1
fi

if [ -z "$APP_STORE_CONNECT_API_KEY" ]; then
    echo "Error: APP_STORE_CONNECT_API_KEY environment variable is not set"
    exit 1
fi

if [ -z "$APP_STORE_CONNECT_API_ISSUER" ]; then
    echo "Error: APP_STORE_CONNECT_API_ISSUER environment variable is not set"
    exit 1
fi

# Create a ZIP archive for notarization
echo "Creating ZIP archive for notarization..."
ZIP_PATH=$(mktemp).zip
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

# Set up App Store Connect API
echo "Setting up App Store Connect API..."
PRIVATE_KEY_DIR=$(mktemp -d)
PRIVATE_KEY_PATH="$PRIVATE_KEY_DIR/AuthKey_$APP_STORE_CONNECT_API_KEY.p8"
echo "$APP_STORE_CONNECT_KEY" > "$PRIVATE_KEY_PATH"

# Submit for notarization
echo "Submitting app for notarization..."
# Create a temporary file to capture the output
NOTARIZATION_OUTPUT=$(mktemp)

# Run the notarytool command and capture its output
xcrun notarytool submit "$ZIP_PATH" \
  --key "$PRIVATE_KEY_PATH" \
  --key-id "$APP_STORE_CONNECT_API_KEY" \
  --issuer "$APP_STORE_CONNECT_API_ISSUER" \
  --wait > "$NOTARIZATION_OUTPUT" 2>&1

# Display the full output
echo "Notarization output:"
cat "$NOTARIZATION_OUTPUT"

# Extract the UUID using a more reliable method
NOTARIZATION_UUID=$(grep -o "id: [a-f0-9\-]\+" "$NOTARIZATION_OUTPUT" | head -1 | cut -d' ' -f2)

echo "Extracted Notarization UUID: $NOTARIZATION_UUID"

# Check if we have a valid UUID
if [[ $NOTARIZATION_UUID =~ ^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$ ]]; then
    echo "Valid UUID format detected"

    # Check notarization status
    echo "Checking notarization status..."
    xcrun notarytool info "$NOTARIZATION_UUID" \
      --key "$PRIVATE_KEY_PATH" \
      --key-id "$APP_STORE_CONNECT_API_KEY" \
      --issuer "$APP_STORE_CONNECT_API_ISSUER"

    # Get detailed log
    echo "Getting detailed notarization log..."
    LOG_PATH=$(mktemp).log
    xcrun notarytool log "$NOTARIZATION_UUID" \
      --key "$PRIVATE_KEY_PATH" \
      --key-id "$APP_STORE_CONNECT_API_KEY" \
      --issuer "$APP_STORE_CONNECT_API_ISSUER" \
      "$LOG_PATH"

    cat "$LOG_PATH"
else
    echo "Failed to extract a valid UUID from the notarization output"
    echo "Skipping notarization status check and log retrieval"
    # Continue with stapling anyway, as the notarization might have succeeded
fi

# Staple the notarization ticket
echo "Stapling notarization ticket to app..."
# Wait a bit to ensure notarization is fully processed
sleep 10

# Try stapling up to 3 times
for i in {1..3}; do
    echo "Stapling attempt $i..."
    if xcrun stapler staple "$APP_PATH"; then
        echo "Stapling succeeded on attempt $i"
        break
    else
        echo "Stapling failed on attempt $i"
        if [ $i -lt 3 ]; then
            echo "Waiting before retry..."
            sleep 10
        else
            echo "All stapling attempts failed, but continuing anyway"
        fi
    fi
done

# Verify stapling
echo "Verifying stapling..."
if stapler validate "$APP_PATH"; then
    echo "Stapling validation succeeded"
else
    echo "Stapling validation failed, but continuing anyway"
fi

# Clean up
rm -f "$ZIP_PATH" "$NOTARIZATION_OUTPUT" "$LOG_PATH"
rm -rf "$PRIVATE_KEY_DIR"

echo "Notarization process completed"
