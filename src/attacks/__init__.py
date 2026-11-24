"""Stage 6: attacks on the evidence ledger."""

# The EvidenceLedger contract's errors and their 4-byte selectors (cast sig), to read a refusal.
ERRORS = {
    "NotOwner": "0x30cd7471",  # NotOwner()
    "NotRouter": "0x2462ed09",  # NotRouter(address)
    "ModelExists": "0x2486e84c",  # ModelExists(bytes32)
    "UnknownModel": "0xa2dc0625",  # UnknownModel(bytes32)
    "BadGroup": "0xf9a074d9",  # BadGroup(uint8)
    "BadScore": "0xe7a47665",  # BadScore(uint16)
    "AlreadyReported": "0x21ef8a71",  # AlreadyReported(bytes32,bytes32)
}
