#!/bin/sh
set -eu

DATA_DIR="${DATA_DIR:-/data}"
BUNDLED_CARDID_DIR="/app/priv/cardid"

mkdir -p "$DATA_DIR"

# Take ownership of a fresh or externally-owned volume once; on normal
# restarts only the directory itself needs checking.
if [ "$(stat -c %U "$DATA_DIR")" != "app" ]; then
  chown -R app:app "$DATA_DIR"
fi

# Seed the bundled beta table detector once. Administrators can still publish a newer CardId
# bundle into DATA_DIR/cardid; an existing current symlink is never replaced at startup.
if [ ! -e "$DATA_DIR/cardid/current" ] && [ -d "$BUNDLED_CARDID_DIR" ]; then
  for bundle in "$BUNDLED_CARDID_DIR"/*; do
    [ -d "$bundle" ] || continue
    version="$(basename "$bundle")"
    mkdir -p "$DATA_DIR/cardid"
    cp -R "$bundle" "$DATA_DIR/cardid/$version"
    ln -s "$version" "$DATA_DIR/cardid/current"
    chown -R app:app "$DATA_DIR/cardid"
    break
  done
fi

if [ -n "${THE_GATHERING_ADMIN_USERNAME:-}" ] || [ -n "${THE_GATHERING_ADMIN_PASSWORD:-}" ]; then
  if [ -z "${THE_GATHERING_ADMIN_USERNAME:-}" ] || [ -z "${THE_GATHERING_ADMIN_PASSWORD:-}" ]; then
    echo "THE_GATHERING_ADMIN_USERNAME and THE_GATHERING_ADMIN_PASSWORD must be set together" >&2
    exit 1
  fi

  su-exec app env -u PHX_SERVER /app/bin/the_gathering eval 'TheGathering.Release.bootstrap_admin()'
fi

exec su-exec app "$@"
