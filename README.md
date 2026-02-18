# Viturka Protocol

A ZKML-native blockchain for decentralized federated learning, where mining is replaced by cryptographically verified AI model training.

Viturka uses **Proof of Credibility (PoC)** — a consensus mechanism where block production is determined by accumulated reputation from validated AI contributions. Validators generate zero-knowledge proofs attesting to correct training execution via **ZKML** (Zero-Knowledge Machine Learning), enabling on-chain verification of both inference and training.

> **Author:** Pratik Save | **Version:** 3.0 — ZKML-Native Architecture | **February 2026**

## What's Built

### Phase 1 — Blockchain Core (Rust)

A complete in-memory blockchain with Proof of Credibility consensus, 77 tests passing.

- **Transaction types:** VIT transfers (Tx), encrypted data contributions (DTx), key reveals (RTx), ZKML validation proofs (PTx)
- **DDTXO lifecycle:** PENDING → REVEALED → VALIDATING → APPROVED/REJECTED with timing windows and consensus thresholds
- **Consensus:** Top-10 validator selection by credibility score, progressive decay with staking reduction, 10-block cooldown
- **Proof pipeline:** Pluggable `ProofSystem` trait — Phase 1 uses `MockProofSystem`, swappable with real ZK backends
- **Validation:** Accuracy checks (NaN/Inf/range), cross-validator consistency (0.001 tolerance), category enforcement

### ZKML Proof-of-Concept (Python/ezkl)

Real zero-knowledge proofs generated and verified using [ezkl](https://github.com/zkonduit/ezkl) v23 + PyTorch + ONNX. No mocks, no stubs — all proofs are cryptographically valid with `check_mode=SAFE`.

#### Verification of Inference (VoI)

Proves a trained model produced specific outputs for given inputs.

| Metric | Value |
|--------|-------|
| Model | Linear(5,8) → ReLU → Linear(8,8) → Sigmoid → Linear(8,2), 138 params |
| Circuit | logrows=12, 601 rows, 1,202 assignments |
| Proving key | 44.26 MB |
| Proving time | 1.56s |
| Proof size | 41.44 KB |
| Verification | **True**, tamper tests PASS |

#### Verification of Training (VoT)

Proves a **single SGD training step** was computed correctly — forward pass, MSE loss, backward pass, and weight update — all inside a ZK circuit.

| Metric | Value |
|--------|-------|
| Circuit | logrows=15, 10,222 rows, 20,444 assignments |
| ONNX graph | 98 nodes, 16 op types |
| Proving key | 156.07 MB |
| Proving time | 3.06s |
| Proof size | 61.88 KB |
| Verification | **True**, tamper tests PASS |

The VoT circuit encodes the entire SGD step as a single ONNX computation graph — manual forward pass, MSE loss gradient, manual backward pass (Sigmoid derivative as `a2*(1-a2)`, ReLU derivative via `Greater+Cast`), and weight update (`W_new = W - 0.01 * grad`). Manual backward matches PyTorch autograd exactly (max diff = 0.00e+00).

### Scaling Reality

Honest assessment of current ZKML limitations:

| Model size | Proving time | pk.key | Feasibility |
|-----------|-------------|--------|-------------|
| 138 params | 3.06s | 156 MB | Current PoC |
| 1,500 params | ~15s | ~500 MB | Practical (CPU) |
| 5,000 params | ~30s | ~1.2 GB | Upper bound (CPU) |
| 50,000 params | ~minutes | ~20 GB | Needs GPU (Icicle) |
| 1M params | hours | ~320 GB | Infeasible today |

**Sweet spot: 1,500-5,000 parameters.** This covers real DeFi use cases — credit scoring (AUC 0.78-0.85), fraud detection (AUC 0.80-0.88), and liquidation risk assessment.

## Project Structure

```
viturka/
├── chain/                    # Core blockchain library (Rust)
│   ├── src/
│   │   ├── types.rs          # Address, Hash, Signature, Keypair, CID
│   │   ├── transaction.rs    # Tx, DTx, RTx, PTx + ProofPublicInputs
│   │   ├── block.rs          # Block, BlockHeader, merkle roots
│   │   ├── consensus.rs      # Top-10 selection, credibility decay, staking
│   │   ├── state.rs          # ChainState, DDTXO lifecycle, block processing
│   │   ├── proof.rs          # ProofSystem trait, MockProofSystem
│   │   ├── error.rs          # Typed errors
│   │   └── lib.rs
│   └── tests/
│       └── integration.rs    # Full lifecycle integration tests
├── cli/                      # Command-line interface
│   └── src/main.rs           # init + demo commands
├── poc/                      # ZKML proof-of-concept (Python)
│   ├── verifiable_training_demo.py   # VoI — inference verification
│   ├── vot_sgdstep_demo.py           # VoT — training step verification
│   ├── artifacts/            # VoI proof artifacts
│   └── vot_artifacts/        # VoT proof artifacts
├── benchmarks/               # Benchmark reports
│   ├── verified_cpu_run.md   # VoI benchmark
│   └── vot_sgdstep_run.md   # VoT benchmark
└── viturka_whitepaper_v3_zkml.md
```

## Building & Running

### Blockchain (Rust)

```bash
# Prerequisites: Rust 1.75+ (stable)
cargo build --workspace

# Initialize a test chain
cargo run -p viturka-cli -- init

# Run full lifecycle demo
cargo run -p viturka-cli -- demo
```

### ZKML PoC (Python)

```bash
# Prerequisites: Python 3.10+
pip install torch ezkl onnx onnxruntime psutil

# Run VoI proof
python poc/verifiable_training_demo.py

# Run VoT proof (SGD step)
python poc/vot_sgdstep_demo.py
```

## Testing

```bash
# All tests (77 total: 69 unit + 8 integration)
cargo test --workspace

# Chain library tests only
cargo test -p viturka-chain

# Integration tests
cargo test -p viturka-chain --test integration
```

| Module | Tests | Coverage |
|--------|-------|----------|
| types | 7 | Hash determinism, address display, keypair signing, merkle roots |
| transaction | 6 | All 4 tx types: serialization, signing, hash stability |
| block | 6 | Genesis creation, merkle roots, block hashing |
| consensus | 18 | Top-10 selection, tie-breaking, cooldown, decay, staking, rewards |
| state | 16 | Full DDTXO lifecycle, transfers, bans, proof rejection |
| proof | 11 | Mock proof generation/verification, accuracy validation |
| integration | 8 | End-to-end lifecycle, rejection, expiration, fake proofs |

## Roadmap

- [x] **Phase 1 — Minimal Chain** — Core types, consensus, DDTXO lifecycle, 77 tests
- [x] **ZKML PoC** — Real ZK proofs for inference (VoI) and training (VoT) with ezkl
- [ ] **Phase 2 — ZKML Integration** — Wire ezkl verification into Rust chain
- [ ] **Phase 3 — Networking & Persistence** — libp2p, RocksDB, block propagation
- [ ] **Phase 4 — Full Protocol** — Validator registration, slashing, federation
- [ ] **Phase 5 — Production** — Testnet, security audit, GPU proving (Icicle)

## License

See [whitepaper](viturka_whitepaper_v3_zkml.md) for full protocol specification.
