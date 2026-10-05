#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
#  Studio Toolbox — builds a native .app shell wrapper + DMG installer
#
#  A real macOS .app bundle with a simple bash launcher inside, NOT a
#  PyInstaller freeze -- PyInstaller kept breaking against a
#  miniconda/tkinter setup. The launcher finds a working Python (the
#  one that actually has torch/cv2/numpy/PIL installed) and runs
#  ai_toolbox.py directly from inside the bundle. No freezing, no
#  bundling of Python itself.
#
#  Place this file in the same folder as ai_toolbox.py.
#  Usage:  chmod +x build_dmg.sh && ./build_dmg.sh
# ─────────────────────────────────────────────────────────────────────

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
# Hyphenated on disk (bundle, executable, DMG) so no path needs quoting
# in Finder, Terminal or the xattr hint below; the display name has the
# space.
APP_NAME="Studio-Toolbox"
DISPLAY_NAME="Studio Toolbox"
VERSION="1.0"
TOOLBOX="$DIR/ai_toolbox.py"
ICON="$DIR/AppIcon.icns"          # optional -- ignored if missing
ICONSET="$DIR/AppIcon.iconset"    # unzip AppIcon.iconset.zip here first

# If a .iconset folder sits here but no .icns yet, build it -- lets a
# freshly generated 1024px icon (or the iconset saved from before) be
# dropped in as-is without a separate manual conversion step.
if [ -d "$ICONSET" ] && [ ! -f "$ICON" ]; then
    echo "Building AppIcon.icns from AppIcon.iconset..."
    iconutil --convert icns "$ICONSET" --output "$ICON"
    echo "AppIcon.icns created"
fi

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║   Studio Toolbox $VERSION  — DMG Builder      ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# ── Preflight ────────────────────────────────────────────────────────
if [ ! -f "$TOOLBOX" ]; then
    echo "ERROR: ai_toolbox.py not found next to this script."
    echo "Expected: $TOOLBOX"
    exit 1
fi
echo "Found ai_toolbox.py ($(wc -l < "$TOOLBOX" | tr -d ' ') lines)"

BUILD="$DIR/dist"
APP="$BUILD/$APP_NAME.app"
rm -rf "$BUILD"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# ── Info.plist ───────────────────────────────────────────────────────
cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundleDisplayName</key><string>$DISPLAY_NAME</string>
    <key>CFBundleIdentifier</key><string>com.aronmakkai.aitoolbox</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

if [ -f "$ICON" ]; then
    cp "$ICON" "$APP/Contents/Resources/AppIcon.icns"
    echo "Icon included"
else
    echo "No AppIcon.icns found -- shipping without a custom icon"
fi

cp "$TOOLBOX" "$APP/Contents/Resources/ai_toolbox.py"

# ── Launcher ─────────────────────────────────────────────────────────
# Tries every Python that could plausibly have this app's real
# dependencies (torch, cv2, numpy, PIL) already installed, in the
# order most likely to be right for a working VFX/ML setup: an active
# conda env first, then miniconda/anaconda's own base env, then
# Homebrew, then whatever "python3" resolves to on PATH. The first one
# that can actually import every dependency wins; if none can, a clear
# dialog says so instead of a silent crash.
cat > "$APP/Contents/MacOS/$APP_NAME" << 'LAUNCHER'
#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"
export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

CANDIDATES=()
[ -n "$CONDA_PREFIX" ] && CANDIDATES+=("$CONDA_PREFIX/bin/python3")
CANDIDATES+=(
    "$HOME/miniconda3/bin/python3"
    "$HOME/anaconda3/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "$(command -v python3 || true)"
)

PYTHON=""
for c in "${CANDIDATES[@]}"; do
    [ -z "$c" ] && continue
    [ -x "$c" ] || continue
    if "$c" -c "import torch, cv2, numpy, PIL" >/dev/null 2>&1; then
        PYTHON="$c"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display dialog "Studio Toolbox could not find a Python install with torch, opencv-python, numpy and Pillow already installed.\n\nActivate the conda environment this app was set up with, then relaunch from Terminal:\n\n  conda activate <your-env>\n  open /Applications/Studio-Toolbox.app" with title "Studio Toolbox — Python not found" buttons {"OK"} default button 1 with icon caution'
    exit 1
fi

exec "$PYTHON" "$DIR/ai_toolbox.py"
LAUNCHER
chmod +x "$APP/Contents/MacOS/$APP_NAME"

echo "App bundle built: $APP"

# ── Signing ──────────────────────────────────────────────────────────
# Two paths, chosen entirely by whether DEVELOPER_ID is set in the
# environment -- nothing here changes behaviour for anyone who hasn't
# set it up, so this stays a no-cost, no-account default until you
# actively opt in.
#
# 1) DEVELOPER_ID unset (default): ad-hoc signing only (codesign
#    accepts "-" as a self-identity, no Apple account needed). This
#    does NOT satisfy Gatekeeper's "identified developer" check, so
#    first launch on another Mac still needs either right-click ->
#    Open -> Open Anyway, or:  xattr -cr /Applications/Studio-Toolbox.app
#    What it DOES fix is a separate, more confusing failure --
#    "Studio Toolbox is damaged and can't be opened" -- which unsigned
#    apps can hit depending on how they were zipped/AirDropped/
#    uploaded, since that process can alter the bundle in ways an
#    unsigned app has no signature to detect.
#
# 2) DEVELOPER_ID set: this is the ONLY way to make Gatekeeper pass
#    with zero warning on someone else's Mac, and it requires actually
#    paying Apple -- there is no free way around this. Needs, one time:
#      - An Apple Developer Program membership ($99/year)
#      - A "Developer ID Application" certificate from that account,
#        installed in this Mac's Keychain (Xcode -> Settings ->
#        Accounts -> Manage Certificates -> +)
#      - An app-specific password for notarization (appleid.apple.com
#        -> Sign-In and Security -> App-Specific Passwords), or a
#        notarytool keychain profile set up once via:
#          xcrun notarytool store-credentials "AC_PASSWORD" \
#              --apple-id "you@example.com" --team-id TEAMID \
#              --password "xxxx-xxxx-xxxx-xxxx"
#    Then run this script as:
#      DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)" \
#      NOTARY_PROFILE="AC_PASSWORD" \
#      ./build_dmg.sh
#    Real signing + notarization + stapling then replaces ad-hoc
#    signing below, and the resulting DMG opens with no warning at all.
if [ -n "$DEVELOPER_ID" ]; then
    echo "Signing with Developer ID: $DEVELOPER_ID"
    codesign --force --deep --options runtime --timestamp \
        --sign "$DEVELOPER_ID" "$APP"
    echo "Signed."
elif command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "$APP" 2>/dev/null \
        && echo "Ad-hoc signed (reduces \"damaged\" false positives; "\
                "does not remove the Gatekeeper unidentified-developer "\
                "warning -- set DEVELOPER_ID to fix that for real, see "\
                "comments above)"
fi

# ── DMG ──────────────────────────────────────────────────────────────
DMG_NAME="$APP_NAME-Installer.dmg"
DMG_STAGE="$BUILD/dmg_stage"
mkdir -p "$DMG_STAGE"
cp -R "$APP" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"

rm -f "$DIR/$DMG_NAME"
hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_STAGE" \
    -ov -format UDZO "$DIR/$DMG_NAME"

# ── Notarization (only runs if DEVELOPER_ID + NOTARY_PROFILE are set)
# Submits the finished DMG to Apple, waits for the result, and staples
# the ticket to it so Gatekeeper can verify it offline afterwards. This
# is the step that actually removes the warning -- signing alone does
# not. Skipped entirely (with a clear note) if you haven't set these up.
if [ -n "$DEVELOPER_ID" ] && [ -n "$NOTARY_PROFILE" ]; then
    echo ""
    echo "Submitting for notarization (this can take a few minutes)…"
    xcrun notarytool submit "$DIR/$DMG_NAME" \
        --keychain-profile "$NOTARY_PROFILE" --wait
    echo "Stapling notarization ticket…"
    xcrun stapler staple "$DIR/$DMG_NAME"
    echo "Notarized and stapled — this DMG opens with NO Gatekeeper warning."
elif [ -n "$DEVELOPER_ID" ]; then
    echo ""
    echo "DEVELOPER_ID set but NOTARY_PROFILE is not -- signed with a"
    echo "real identity, but NOT notarized, so Gatekeeper will still"
    echo "warn once. Set NOTARY_PROFILE too (see comments above) to"
    echo "remove the warning completely."
fi

echo ""
echo "Done — $DMG_NAME"
echo "    Open DMG → drag Studio-Toolbox to Applications → launch"
echo ""
open "$DIR"
