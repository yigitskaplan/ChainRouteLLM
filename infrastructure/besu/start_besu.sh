#!/bin/bash
# Single-validator Hyperledger Besu network with QBFT consensus (2 s blocks, zero base fee), used by
# contracts/measure_ledger.py. Requires Docker. First run creates the genesis, the validator key and a
# throw-away development account (dev_wallet.json, never used anywhere else).
set -e
cd "$(dirname "$0")"
IMG=hyperledger/besu:26.11.0
export PATH=~/.foundry/bin:$PATH
[ -f dev_wallet.json ] || cast wallet new --json > dev_wallet.json
ADDR=$(python3 -c "import json;print(json.load(open('dev_wallet.json'))['data'][0]['address'])")
if [ ! -f data/networkFiles/genesis.json ]; then
  sed "s/\"alloc\": {\"[^\"]*\"/\"alloc\": {\"${ADDR#0x}\"/" data/qbftConfigFile.json > data/qbft.json
  docker run --rm -v "$PWD/data:/data" $IMG operator generate-blockchain-config \
    --config-file=/data/qbft.json --to=/data/networkFiles --private-key-file-name=key
fi
mkdir -p data/node && cp data/networkFiles/keys/*/key data/node/key
docker rm -f besu-p5 >/dev/null 2>&1 || true
docker run -d --name besu-p5 -p 127.0.0.1:8560:8545 -v "$PWD/data:/data" $IMG \
  --data-path=/data/node --genesis-file=/data/networkFiles/genesis.json --rpc-http-enabled --rpc-http-host=0.0.0.0 \
  --rpc-http-api=ETH,NET,QBFT,WEB3,TXPOOL --host-allowlist="*" --min-gas-price=0 --profile=ENTERPRISE
echo "Besu QBFT node on http://127.0.0.1:8560 (dev account $ADDR)"
