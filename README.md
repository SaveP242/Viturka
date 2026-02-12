# Viturka Protocol

A credibility-based blockchain for decentralized federated learning, where mining is replaced by validated AI model training.

Viturka uses **Proof of Credibility (PoC)** — a consensus mechanism where block production is determined by accumulated reputation from validated AI contributions, not hash power or financial stake. Validators generate zero-knowledge proofs attesting to correct training execution via **ZKML** (Zero-Knowledge Machine Learning), enabling cryptographic verification of model training on-chain.

> **Author:** Pratik Save | **Version:** 3.0 — ZKML-Native Architecture | **February 2026**

## Architecture

```
viturka/
├── chain/          # Core blockchain library
│   ├── src/
│   │   ├── types.rs         # Address, Hash, Signature, Keypair, CID
│   │   ├── transaction.rs   # Tx, DTx, RTx, PTx + ProofPublicInputs
│   │   ├── block.rs         # Block, BlockHeader, merkle roots
│   │   ├── consensus.rs     # Top-10 selection, credibility decay, staking
│   │   ├── state.rs         # ChainState, DDTXO lifecycle, block processing
│   │   ├── proof.rs         # ProofSystem trait, MockProofSystem, accuracy validation
│   │   ├── error.rs         # Typed errors (ChainError)
│   │   └── lib.rs           # Module re-exports
│   └── tests/
│       └── integration.rs   # Full lifecycle integration tests
├── cli/            # Command-line interface
│   └── src/
│       └── main.rs          # init + demo commands
└── viturka_whitepaper_v3_zkml.md
```

## Core Concepts

### Transaction Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| **Tx** | VIT token transfer | sender, recipient, amount, fee |
| **DTx** | Data contribution (commit phase) | data_hash, encrypted_cid, key_hash, 100 VIT deposit |
| **RTx** | Reveal decryption key | ddtxo_reference, decryption_key (verified against commitment) |
| **PTx** | ZKML validation proof | validator, proof_bytes, public_inputs, new_weights_cid |

### DDTXO Lifecycle

```
DTx submitted          RTx reveals key       7+ PTx proofs         Deadline reached
     │                      │                     │                      │
  PENDING ──[10 blocks]── REVEALED ──[proofs]── VALIDATING ──[check]── APPROVED
     │                      │                                           │
     └── EXPIRED            └── INVALID                              REJECTED
         (no reveal)            (wrong key → ban)                  (no improvement)
```

- **Reveal window:** 10 blocks after DTx inclusion
- **Validation window:** 15 blocks after reveal deadline
- **Consensus threshold:** 7 of 10 top validators must submit valid proofs
- **Wrong key reveal:** Contributor permanently banned, fee forfeited

### Proof of Credibility

Validators are ranked by credibility score. The **top 10** are eligible for each validation round.

- **Earning credibility:** Approved data contributions (+10-20), valid proofs (+5), block proposals (+20)
- **Progressive decay:** Higher credibility decays faster — `rate = 3% × (cred/10000)^1.5` per month
- **Staking reduction:** Up to 50% decay reduction at 5,000 VIT staked
- **Cooldown:** 10-block halt period after participating in validation
- **Tie-breaking:** Address ascending (deterministic)

### Proof Verification Pipeline

The `ProofSystem` trait abstracts ZK proof backends:

```rust
pub trait ProofSystem: Send + Sync + Debug {
    fn generate_proof(&self, pk: &ProvingKey, validator: &Address, inputs: &ProofPublicInputs) -> Vec<u8>;
    fn verify_proof(&self, vk: &VerificationKey, proof: &[u8], validator: &Address, inputs: &ProofPublicInputs) -> bool;
    fn derive_verification_key(&self, pk: &ProvingKey) -> VerificationKey;
}
```

**Phase 1** uses `MockProofSystem` (SHA-256 hash-based, deterministic).
**Phase 2** will swap in real EZKL/Halo2 ZK-SNARK verification — no changes needed to state, consensus, or DDTXO logic.

Verification enforces:
- Proof bound to specific validator identity (no replay)
- Proof bound to exact public inputs (no tampering)
- Accuracy values validated (NaN/Inf/range rejection)
- Cross-validator consistency (0.001 tolerance, deterministic ZKML)
- Category enforcement (validator must support the model category)

## Building

```bash
# Prerequisites: Rust 1.75+ (stable toolchain)
cargo build --workspace
```

## Running

```bash
# Initialize a test chain with genesis block
cargo run -p viturka-cli -- init

# Run the full lifecycle demo
cargo run -p viturka-cli -- demo
```

The `demo` command walks through the complete data contribution lifecycle:
1. Contributor submits encrypted data (DTx) with 100 VIT deposit
2. 10 blocks pass (waiting period)
3. Contributor reveals decryption key (RTx), verified against hash commitment
4. 8 validators submit ZKML proofs (PTx) with accuracy claims
5. Chain verifies proofs, checks accuracy consistency and improvement
6. DDTXO finalized as APPROVED — contributor gets fee refund + reward
7. Participating validators enter cooldown, model checkpoint updated

## Testing

```bash
# Run all tests (77 total: 69 unit + 8 integration)
cargo test --workspace

# Run only chain library tests
cargo test -p viturka-chain

# Run integration tests
cargo test -p viturka-chain --test integration
```

### Test Coverage

| Module | Tests | What's Covered |
|--------|-------|----------------|
| `types` | 7 | Hash determinism, address display, keypair signing, merkle roots |
| `transaction` | 6 | All 4 tx types: serialization, signing, hash stability |
| `block` | 6 | Genesis creation, merkle roots, block hashing |
| `consensus` | 18 | Top-10 selection, tie-breaking, cooldown, decay, staking, rewards |
| `state` | 16 | Full DDTXO lifecycle, transfers, bans, proof rejection, NaN, categories |
| `proof` | 11 | Mock proof generation/verification, accuracy validation, consistency |
| `integration` | 8 | End-to-end lifecycle, rejection, expiration, fake proofs |

## Protocol Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOP_VALIDATORS` | 10 | Max validators per round |
| `CONSENSUS_THRESHOLD` | 7 | Min valid proofs for approval |
| `HALT_PERIOD` | 10 blocks | Cooldown after validation |
| `MIN_CREDIBILITY` | 1,000 | Minimum to be eligible |
| `REVEAL_WINDOW` | 10 blocks | Time to reveal decryption key |
| `VALIDATION_WINDOW` | 15 blocks | Time for validators to submit proofs |
| `DTX_DEPOSIT` | 100 VIT | Required fee deposit |
| `BASE_BLOCK_REWARD` | 50 VIT | Block proposer reward |
| `BASE_DATA_REWARD` | 50 VIT | Contributor reward (base) |
| `BASE_VALIDATION_REWARD` | 10 VIT | Per-validator proof reward |
| `MIN_ACCURACY_IMPROVEMENT` | 0.1% | Minimum improvement for approval |

## Roadmap

- [x] **Phase 1 — Minimal Chain** (Complete)
  - Core types, transactions, blocks, in-memory state
  - DDTXO state machine with full lifecycle
  - Proof of Credibility consensus with top-10 selection
  - Pluggable ProofSystem trait with MockProofSystem
  - Credibility decay with staking reduction
  - Accuracy validation and cross-validator consistency
  - CLI with init and demo commands
  - 77 tests passing

- [ ] **Phase 2 — ZKML Integration**
  - EZKL/Halo2 ZK-SNARK proof verification
  - ONNX model compilation to ZK circuits
  - Real proving key / verification key generation
  - Benchmark: proof generation and verification times

- [ ] **Phase 3 — Networking & Persistence**
  - libp2p peer-to-peer networking
  - RocksDB persistent state storage
  - Block propagation and transaction gossip
  - Multi-node consensus

- [ ] **Phase 4 — Full Protocol**
  - Validator registration and staking
  - Slashing for misbehavior
  - Dynamic category management
  - Cross-category model federation

- [ ] **Phase 5 — Production**
  - Testnet launch
  - Security audit
  - Performance optimization
  - GPU-accelerated proving (Icicle)

## License

See whitepaper for full protocol specification.
