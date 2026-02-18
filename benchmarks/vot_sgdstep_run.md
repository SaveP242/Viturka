# VoT SGDStep - Benchmark Report

**Generated**: 2026-02-16 03:08:47
**Platform**: Windows (AMD64)
**CPU**: AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD
**Python**: 3.10.0
**Packages**: torch=2.10.0+cpu, ezkl=23.0.3

## What This Proves

This circuit proves a **single SGD training step** was computed correctly:
- Given weights W_before, training sample (x, y)
- Compute: forward -> MSE loss -> backward -> weight update
- Output: W_after = W_before - 0.01 * gradient

This is **Verification of Training (VoT)**, not just inference.

## Model

| Property | Value |
|----------|-------|
| Architecture | Linear(5,8)->ReLU->Linear(8,8)->Sigmoid->Linear(8,2) |
| Parameters | 138 |
| Loss | MSE (L = 0.5 * norm(output - target)^2) |
| Optimizer | SGD, lr=0.01 (hardcoded) |
| Batch size | 1 |
| ONNX ops | Add, Cast, Concat, Constant, Gather, Greater, MatMul, Mul, Relu, Reshape, Sigmoid, Slice, Squeeze, Sub, Transpose, Unsqueeze |
| ONNX nodes | 98 |

## Circuit

| Property | Value |
|----------|-------|
| logrows | **15** (2^15 = 32768 rows) |
| num_rows | 10222 |
| total_assignments | 20,444 |
| utilization | 10222/32768 = 31.2% |
| input_scale | 11 |
| param_scale | 11 |
| decomp_base | 16384 |
| decomp_legs | 2 |
| check_mode | **SAFE** |
| Input visibility | public |
| Output visibility | public |
| required_lookups | [{'Sigmoid': {'scale': 2048.0}}] |
| range_checks | [[0, 16383], [-1, 1]] |
| lookup_range | [-14950, 15768] |

## VoI vs VoT Comparison

| Metric | VoI (inference) | VoT (training step) |
|--------|-----------------|---------------------|
| logrows | 12 | 15 |
| num_rows | 601 | 10222 |
| total_assignments | 1,202 | 20,444 |
| pk.key | 44.26 MB | 156.07 MB |
| Proving time | 1.56s | 3.06s |
| Proof size | 41.44 KB | 61.88 KB |
| ONNX nodes | ~20 | 98 |

## Security

| Check | Status |
|-------|--------|
| check_mode | **SAFE** |
| SRS source | LOCAL (gen_srs) |
| Tamper (proof bytes) | **PASS** |
| Tamper (instances) | **PASS** |

## Performance

| Metric | Value |
|--------|-------|
| Proving time | 3.06s |
| Verify time | 0.07s |
| Peak RSS | 222.30 MB |
| RSS delta | -0.40 MB |
| pk.key | 156.07 MB |
| vk.key | 73.51 KB |
| Proof size | 61.88 KB |
| ezkl.verify() | **True** |

## Artifacts

| File | Size |
|------|------|
| `sgd_step.onnx` | 7.92 KB |
| `model.compiled` | 14.42 KB |
| `kzg.srs` | 4096.25 KB |
| `pk.key` | 156.07 MB |
| `vk.key` | 73.51 KB |
| `witness.json` | 42.26 KB |
| `proof.json` | 61.88 KB |
| `settings.json` | 3.50 KB |

## Verification

```
ezkl.verify() returned: True
check_mode: SAFE
Tamper test (proof bytes): PASS
Tamper test (instances): PASS
```

All artifacts are real. check_mode=SAFE. No mocks, no stubs.
This is a real ZK proof of a real SGD training step.
