#!/bin/bash
# Stop the 4 Besu validators of start_besu.sh (Docker project "chainroutellm"). With --wipe, also
# delete the chain, its keys and the development account, so the next start creates a new chain.
set -euo pipefail
cd "$(dirname "$0")"
docker compose down
if [ "${1:-}" = "--wipe" ]; then
  rm -rf data dev_wallet.json
  echo "stopped and wiped"
else
  echo "stopped (chain kept in infrastructure/besu/data)"
fi
