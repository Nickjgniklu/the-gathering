#!/bin/sh
set -eu

DATA_DIR="${DATA_DIR:-/data}"

mkdir -p "$DATA_DIR"

# Take ownership of a fresh or externally-owned volume once; on normal
# restarts only the directory itself needs checking.
if [ "$(stat -c %U "$DATA_DIR")" != "app" ]; then
  chown -R app:app "$DATA_DIR"
fi

if [ -n "${THE_GATHERING_ADMIN_USERNAME:-}" ] || [ -n "${THE_GATHERING_ADMIN_PASSWORD:-}" ]; then
  if [ -z "${THE_GATHERING_ADMIN_USERNAME:-}" ] || [ -z "${THE_GATHERING_ADMIN_PASSWORD:-}" ]; then
    echo "THE_GATHERING_ADMIN_USERNAME and THE_GATHERING_ADMIN_PASSWORD must be set together" >&2
    exit 1
  fi

  su-exec app env -u PHX_SERVER /app/bin/the_gathering eval 'TheGathering.Release.bootstrap_admin()'
fi

exec su-exec app "$@"
