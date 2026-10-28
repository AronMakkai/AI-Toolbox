#!/bin/bash
# Studio Toolbox -- clears the quarantine flag macOS puts on a
# downloaded, unsigned app, makes the launcher executable, and opens it.
APP="/Applications/Studio-Toolbox.app"
if [ ! -d "$APP" ]; then
    echo "Studio-Toolbox.app is not in /Applications yet."
    echo "Drag it there from the installer window first, then run this again."
    read -n 1 -s -r -p "Press any key to close."
    exit 1
fi
echo "Clearing quarantine on $APP ..."
xattr -cr "$APP"
chmod +x "$APP/Contents/MacOS/Studio-Toolbox"
echo "Done. Opening Studio Toolbox..."
open "$APP"
