// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Evidence ledger for LLM routing. Three ways to record a routing outcome are implemented so that
/// their gas cost can be compared: (1) full record in contract storage, (2) record emitted as an event,
/// (3) a batch of records committed as a single Merkle root (records stay off-chain).
contract EvidenceLedger {
    struct Record {
        bytes32 model;      // model identifier
        uint8 task;         // task type predicted by the router
        uint16 qualityBps;  // reported quality, basis points
        uint64 costNanoUsd; // cost of the call
        uint32 latencyMs;
        address reporter;
        bytes32 receipt;    // router-signed request receipt
    }

    address public owner;
    mapping(address => bool) public reporter;
    Record[] public records;
    bytes32[] public roots;
    uint32[] public rootSizes;

    event Evidence(bytes32 indexed model, address indexed reporter, uint8 task, uint16 qualityBps,
                   uint64 costNanoUsd, uint32 latencyMs, bytes32 receipt);
    event BatchCommitted(uint256 indexed batchId, bytes32 root, uint32 count);

    constructor() { owner = msg.sender; reporter[msg.sender] = true; }

    modifier onlyReporter() { require(reporter[msg.sender], "not a reporter"); _; }

    function addReporter(address r) external { require(msg.sender == owner, "owner"); reporter[r] = true; }

    function storeRecord(bytes32 model, uint8 task, uint16 q, uint64 c, uint32 l, bytes32 receipt) external onlyReporter {
        records.push(Record(model, task, q, c, l, msg.sender, receipt));
    }

    function logRecord(bytes32 model, uint8 task, uint16 q, uint64 c, uint32 l, bytes32 receipt) external onlyReporter {
        emit Evidence(model, msg.sender, task, q, c, l, receipt);
    }

    function commitBatch(bytes32 root, uint32 count) external onlyReporter returns (uint256 id) {
        id = roots.length;
        roots.push(root);
        rootSizes.push(count);
        emit BatchCommitted(id, root, count);
    }

    /// Checks that `leaf` is the `index`-th record of batch `batchId` (sha256 Merkle tree, power-of-two batches).
    function verify(uint256 batchId, bytes32 leaf, bytes32[] calldata proof, uint256 index) external view returns (bool) {
        bytes32 h = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            if (index & 1 == 0) h = sha256(abi.encodePacked(h, proof[i]));
            else h = sha256(abi.encodePacked(proof[i], h));
            index >>= 1;
        }
        return h == roots[batchId];
    }

    function recordCount() external view returns (uint256) { return records.length; }
}
