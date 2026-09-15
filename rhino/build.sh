#!/bin/sh
# Build the SitePlan Rhino plugin (.rhp + .yak) headlessly from SitePlan.rhproj.
#
# Why the symlink dance: rhinocode's shell wrapper expands $@ unquoted, so any
# space in the project path ("Site Plan Drafter") splits the argument and the
# build dies with "File does not exist". Building through a space-free symlink
# sidesteps it; output still lands in rhino/build/ here.
#
# Version: pass e.g. `./build.sh 0.2.0` (defaults to the .rhproj's version).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
RHINOCODE="/Applications/Rhino 8.app/Contents/Resources/bin/rhinocode"
LINK="${TMPDIR:-/tmp}/siteplan-rhproj-link"

ln -sfn "$HERE" "$LINK"
if [ -n "$1" ]; then
  "$RHINOCODE" project build "$LINK/SitePlan.rhproj" --buildversion "$1"
else
  "$RHINOCODE" project build "$LINK/SitePlan.rhproj"
fi
rm -f "$LINK"
echo "Artifacts in: $HERE/build/rh8/ (.rhp, .rui, .yak)"
