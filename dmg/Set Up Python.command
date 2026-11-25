#!/bin/bash
# One-time Python setup for Studio Toolbox (same as the dialog the app
# shows when it finds no Python). Works from the installer window or
# after the app is in /Applications.
HERE="$(cd "$(dirname "$0")" && pwd)"
for APP in "/Applications/Studio-Toolbox.app" "$HERE/Studio-Toolbox.app"; do
    S="$APP/Contents/Resources/setup_python.sh"
    if [ -f "$S" ]; then
        exec /bin/bash "$S" "$APP"
    fi
done
echo "Could not find Studio-Toolbox.app next to this file or in /Applications."
read -r -p "Press Return to close." _
