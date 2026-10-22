#!/usr/bin/env bash
# Compatibility command; the archived worker still runs from the repository root.
exec bash "$(dirname "${BASH_SOURCE[0]}")/scripts/archive/download_egotouch_worker.sh" "$@"
