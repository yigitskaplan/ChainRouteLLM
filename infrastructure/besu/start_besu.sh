#!/bin/bash
# Stage 5: a private Hyperledger Besu network of 4 validators with QBFT consensus, in Docker
# (project "chainroutellm", see docker-compose.yml).
#
#   infrastructure/besu/start_besu.sh            # first run creates the chain, later runs restart it
#   infrastructure/besu/stop_besu.sh [--wipe]    # stop; --wipe also deletes the chain and its keys
#
# 4 is the smallest QBFT network that keeps working with one crashed or dishonest validator
# (n >= 3f + 1): a block needs ceil(2n/3) = 3 of the 4 signatures. Blocks every 2 s, gas price 0.
# Based on Fevzi Ege Yurtsevenler's single-validator script (phase5 branch).
#
# The first run creates a throw-away development account (dev_wallet.json, local tests only,
# funded in the genesis), the genesis file and the 4 validator keys (data/networkFiles/).
# Everything generated stays out of Git. Requires Docker and Foundry (cast).
# JSON-RPC: validator 1 on http://127.0.0.1:8545, validators 2-4 on ports 8546-8548.
set -euo pipefail
cd "$(dirname "$0")"
IMG=hyperledger/besu:26.11.0
PREFIX=172.28.0  # validator i listens on $PREFIX.1i:30303 (docker-compose.yml)
export PATH="$HOME/.foundry/bin:$PATH"

[ -f dev_wallet.json ] || cast wallet new --json > dev_wallet.json
ADDR=$(python3 -c "import json; print(json.load(open('dev_wallet.json'))['data'][0]['address'])")

if [ ! -f data/networkFiles/genesis.json ]; then
  mkdir -p data
  # Fund the development account in the genesis, then let Besu create the genesis and the keys.
  python3 - "$ADDR" <<'EOF'
import json, sys
config = json.load(open("qbftConfigFile.json"))
config["genesis"]["alloc"] = {sys.argv[1].lower().removeprefix("0x"): {"balance": "0xad78ebc5ac6200000"}}
json.dump(config, open("data/qbft.json", "w"), indent=2)
EOF
  # Besu reports "Output directory already exists" after writing the files; the check below decides.
  docker run --rm -v "$PWD/data:/data" "$IMG" operator generate-blockchain-config \
    --config-file=/data/qbft.json --to=/data/networkFiles --private-key-file-name=key || true
  [ -f data/networkFiles/genesis.json ] || { echo "genesis was not created"; exit 1; }
fi

KEYS=(data/networkFiles/keys/*)  # one folder per validator, named by its address
[ "${#KEYS[@]}" -eq 4 ] || { echo "expected 4 validator keys, found ${#KEYS[@]}"; exit 1; }
ENODES=()
for i in 1 2 3 4; do ENODES+=("enode://$(sed 's/^0x//' "${KEYS[$((i - 1))]}/key.pub")@$PREFIX.1$i:30303"); done

for i in 1 2 3 4; do
  mkdir -p "data/node$i"
  cp "${KEYS[$((i - 1))]}/key" "data/node$i/key"
  # Every validator keeps a permanent link to the other three (static-nodes.json in its data
  # folder), so the network has no centre and survives the loss of any one validator.
  python3 -c "import json, sys; json.dump(sys.argv[2:], open(sys.argv[1], 'w'), indent=2)" \
    "data/node$i/static-nodes.json" $(for j in 1 2 3 4; do [ "$j" -ne "$i" ] && echo "${ENODES[$((j - 1))]}"; done)
done

docker compose up -d --quiet-pull
echo "4 Besu QBFT validators started: http://127.0.0.1:8545-8548 (dev account $ADDR)"
