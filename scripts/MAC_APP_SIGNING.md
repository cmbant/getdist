# Mac App Signing and Notarization Guide

This document explains how to set up code signing and notarization for the GetDist GUI Mac app in GitHub Actions.

## Prerequisites

To sign and notarize the Mac app, you need:

1. An Apple Developer account (paid membership)
2. A Developer ID Application certificate
3. App-specific password for your Apple ID

## Setting Up GitHub Secrets

The following secrets need to be added to your GitHub repository:

| Secret Name | Description |
|-------------|-------------|
| `MACOS_CERTIFICATE` | Base64-encoded Developer ID Application certificate (p12 format) |
| `MACOS_CERTIFICATE_PWD` | Password for the certificate |
| `APP_STORE_CONNECT_API_ISSUER` | App Store Connect API Issuer ID (found in App Store Connect > Users and Access > Keys) |
| `APP_STORE_CONNECT_API_KEY` | App Store Connect API Key ID (e.g., "DEADBEEF") |
| `APP_STORE_CONNECT_KEY` | The contents of the App Store Connect API private key file (.p8) |

The workflow uses native macOS commands and tools (`codesign`, `notarytool`, etc.) to handle the signing and notarization process, which provides direct control over the process and avoids potential issues with third-party actions.

## Hardened Runtime

The hardened runtime is a security feature required by Apple for notarization. It helps protect your app by:

- Preventing code injection
- Restricting access to system resources
- Enforcing code signature validation

The workflow creates an entitlements file with the following exceptions to ensure the app functions correctly:

- `com.apple.security.cs.allow-jit`: Allows just-in-time compilation
- `com.apple.security.cs.allow-unsigned-executable-memory`: Allows unsigned executable memory
- `com.apple.security.cs.disable-library-validation`: Allows loading unsigned libraries
- `com.apple.security.cs.allow-dyld-environment-variables`: Allows dynamic loader environment variables
- `com.apple.security.automation.apple-events`: Allows sending Apple events

### Creating and Exporting the Certificate

1. Request a Developer ID Application certificate from Apple Developer portal
2. Export the certificate from Keychain Access as a p12 file with a password
3. Convert the p12 file to base64:

   ```bash
   base64 -i path/to/certificate.p12 | pbcopy
   ```

4. Paste the base64-encoded certificate into the `MACOS_CERTIFICATE` secret

### Creating App Store Connect API Keys

1. Go to [App Store Connect](https://appstoreconnect.apple.com/)
2. Sign in with your Apple ID
3. Go to "Users and Access" > "Keys"
4. Click the "+" button to create a new key
5. Give it a name (e.g., "GitHub Actions")
6. Select the "Developer" role
7. Click "Generate"
8. Download the API key (.p8 file)
9. Note the Key ID and Issuer ID
10. Save the Key ID as `APP_STORE_CONNECT_API_KEY`
11. Save the Issuer ID as `APP_STORE_CONNECT_API_ISSUER`
12. Save the contents of the .p8 file as `APP_STORE_CONNECT_KEY`

The .p8 file contents can be viewed with a text editor. It should look something like:
```
-----BEGIN PRIVATE KEY-----
MIGTAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBHkwdwIBAQQgU4Xp1UUQ/6Xj
...
KyQIyRvbXJTKUwIBBQ==
-----END PRIVATE KEY-----
```

## Manually Triggering a Build

To manually trigger a build:

1. Go to the GitHub repository
2. Click on the "Actions" tab
3. Select the "Build Mac App" workflow
4. Click "Run workflow"
5. By default, the app will be signed and notarized if the required secrets are set up
6. If you want to skip signing (for testing purposes), check the "Skip signing and notarization" option
7. Click "Run workflow"

## Automatic Signing for Builds

The workflow will automatically sign and notarize the app for:

1. Any push to the main/master/mac branches that affects the relevant files
2. Any manually triggered workflow (unless signing is explicitly skipped)
3. Any tagged release

For tagged releases, the signed and notarized DMG will also be attached to the GitHub release.

## Verifying the Signed App

To verify that the app is properly signed and notarized:

1. Download the DMG from the GitHub Actions artifacts or release
2. Mount the DMG and open the app
3. macOS should open the app without any security warnings

You can also verify the signature manually:

```bash
codesign --verify --verbose "GetDist GUI.app"
spctl --assess --verbose "GetDist GUI.app"
```

## Troubleshooting

### Certificate Issues

If you see errors related to the certificate:

1. Make sure the certificate is not expired
2. Verify that the base64 encoding is correct
3. Check that the certificate password is correct
4. Ensure your Apple Developer account has the necessary permissions

### App Store Connect API Issues

If you see errors related to the App Store Connect API:

1. Verify that the API key is not expired (they expire after 1 year)
2. Check that the Issuer ID is correct
3. Ensure the Key ID matches the .p8 file you're using
4. Verify that the API key has the necessary permissions (Developer role)
5. Make sure the .p8 file contents are correctly formatted with the BEGIN and END markers

### Path and Filename Issues

If you see errors related to file paths or filenames:

1. Avoid spaces in app bundle names and paths when using with signing tools
2. The workflow automatically renames the app bundle to remove spaces before signing
3. If you see "File name too long" errors, check that the renaming step is working correctly
4. For custom app names, ensure they don't contain special characters that might cause issues

### Notarization Issues

If notarization fails:

1. Check the workflow logs for detailed error messages
2. Ensure the app bundle meets Apple's notarization requirements:
   - All executables must be signed with a Developer ID certificate
   - The app must have a secure timestamp
   - The app must opt into the Hardened Runtime (this is now automatically enabled in our workflow)
   - The app must not contain any malicious code
3. Try running the notarization process manually on a Mac to get more detailed error messages

## References

- [Apple Code Signing Documentation](https://developer.apple.com/documentation/security/code_signing)
- [Notarizing macOS Software](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [GitHub Actions for macOS](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources)
