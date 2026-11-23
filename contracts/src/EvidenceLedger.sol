// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title EvidenceLedger
/// @notice Stage 5 ledger of LLM routing evidence. For every routed question an approved router adds
/// one receipt: the question's keccak256 hash, its automatic BERT group, the model, the score, the
/// cost and the latency. No question text and no model answer is ever stored. The reporter (the
/// sender) and the time (the block) come with every transaction. Receipts are events, so every
/// router can read all of them and add up its own totals per (group, model).
/// @dev Based on Fevzi Ege Yurtsevenler's EvidenceLedger (phase5 branch): approved reporters and the
/// event-based record. A research prototype: one owner approves routers and registers models.
contract EvidenceLedger {
    /// Automatic question groups: k-means with K = 8 on the BERT vectors of the questions.
    uint8 public constant GROUPS = 8;
    /// Scores are in basis points: 10,000 = 100.00%, so partial scores fit as well as right/wrong.
    uint16 public constant MAX_SCORE = 10_000;

    struct Model {
        bool registered;
        uint64 priceIn;      // official price, micro-USD per million input tokens ($4 = 4,000,000)
        uint64 priceOut;     // official price, micro-USD per million output tokens
        uint64 registeredAt; // block time of registration: a new model starts from here
        string name;
    }

    address public immutable owner;
    mapping(address => bool) public isRouter;
    mapping(bytes32 => Model) public models;   // model id = keccak256(name)
    mapping(bytes32 => bool) public reported;  // keccak256(router, question hash, model id)

    event RouterSet(address indexed router, bool approved);
    event ModelRegistered(bytes32 indexed modelId, string name, uint64 priceIn, uint64 priceOut);
    event Receipt(
        bytes32 indexed questionHash,
        bytes32 indexed modelId,
        address indexed router,
        uint8 group,
        uint16 score,
        uint64 costNanoUsd,
        uint32 latencyMs
    );

    error NotOwner();
    error NotRouter(address sender);
    error ModelExists(bytes32 modelId);
    error UnknownModel(bytes32 modelId);
    error BadGroup(uint8 group);
    error BadScore(uint16 score);
    error AlreadyReported(bytes32 questionHash, bytes32 modelId);

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addRouter(address router) external onlyOwner {
        isRouter[router] = true;
        emit RouterSet(router, true);
    }

    function removeRouter(address router) external onlyOwner {
        isRouter[router] = false;
        emit RouterSet(router, false);
    }

    /// Registers a model with its official prices; receipts are accepted only for registered models.
    function registerModel(string calldata name, uint64 priceIn, uint64 priceOut) external onlyOwner returns (bytes32 id) {
        id = keccak256(bytes(name));
        if (models[id].registered) revert ModelExists(id);
        // casting to 'uint64' is safe because block times in seconds fit in 64 bits for billions of years
        // forge-lint: disable-next-line(unsafe-typecast)
        models[id] = Model(true, priceIn, priceOut, uint64(block.timestamp), name);
        emit ModelRegistered(id, name, priceIn, priceOut);
    }

    /// Adds one routing receipt. A router may report a question only once per model.
    function addReceipt(
        bytes32 questionHash,
        uint8 group,
        bytes32 modelId,
        uint16 score,
        uint64 costNanoUsd,
        uint32 latencyMs
    ) external {
        if (!isRouter[msg.sender]) revert NotRouter(msg.sender);
        if (!models[modelId].registered) revert UnknownModel(modelId);
        if (group >= GROUPS) revert BadGroup(group);
        if (score > MAX_SCORE) revert BadScore(score);
        bytes32 key = keccak256(abi.encode(msg.sender, questionHash, modelId));
        if (reported[key]) revert AlreadyReported(questionHash, modelId);
        reported[key] = true;
        emit Receipt(questionHash, modelId, msg.sender, group, score, costNanoUsd, latencyMs);
    }

    /// The id a model gets when it is registered under `name`.
    function modelIdOf(string calldata name) external pure returns (bytes32) {
        return keccak256(bytes(name));
    }
}
