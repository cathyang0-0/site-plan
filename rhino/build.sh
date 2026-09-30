#!/bin/sh
# Build the SitePlan Rhino plugin (.rhp + .yak). Thin wrapper over build.py,
# which bundles the Libraries/ modules into the command at build time (see
# its docstring for why) and runs rhinocode from a space-free staging dir.
#
# Version: pass e.g. `./build.sh 0.2.0` (defaults to the .rhproj's version).
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/build.py" "$@"
