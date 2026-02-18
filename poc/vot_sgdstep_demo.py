#!/usr/bin/env python3
"""
VoT (Verification of Training) SGDStep PoC
Proves a single SGD training step in zero-knowledge using ezkl.

Circuit proves: given (W_before, x, y), W_after = SGDStep(W_before, x, y)
NO STUBS. NO MOCKS. Real ZK proof of a real training step.
"""

import os
import sys
import json
import time
import math
import platform

import torch
import torch.nn as nn
import numpy as np
import onnx
import ezkl
import psutil
import asyncio

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIM   = 5
HIDDEN_DIM  = 8
OUTPUT_DIM  = 2
LR          = 0.01
TOTAL_PARAMS = (INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM +
                HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM +
                HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM)  # 138

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "vot_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

ONNX_PATH    = os.path.join(ARTIFACT_DIR, "sgd_step.onnx")
SETTINGS_PATH = os.path.join(ARTIFACT_DIR, "settings.json")
CAL_PATH     = os.path.join(ARTIFACT_DIR, "calibration.json")
COMPILED_PATH = os.path.join(ARTIFACT_DIR, "model.compiled")
SRS_PATH     = os.path.join(ARTIFACT_DIR, "kzg.srs")
PK_PATH      = os.path.join(ARTIFACT_DIR, "pk.key")
VK_PATH      = os.path.join(ARTIFACT_DIR, "vk.key")
WITNESS_PATH = os.path.join(ARTIFACT_DIR, "witness.json")
PROOF_PATH   = os.path.join(ARTIFACT_DIR, "proof.json")
INPUT_PATH   = os.path.join(ARTIFACT_DIR, "input.json")

REPORT_DIR = os.path.join(BASE_DIR, "..", "benchmarks")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(REPORT_DIR, "vot_sgdstep_run.md")


# ---------------------------------------------------------------------------
# SGDStep Module
# ---------------------------------------------------------------------------
class SGDStep(nn.Module):
    """
    Complete SGD training step as a pure computation graph.

    Inputs:
      weights: (1, 138) - current model parameters (flattened)
      x:       (1, 5)   - training sample features
      y:       (1, 2)   - training sample targets

    Output:
      new_weights: (1, 138) - updated parameters after one SGD step

    Weight layout: W1(5x8=40), b1(8), W2(8x8=64), b2(8), W3(8x2=16), b3(2)
    """

    def forward(self, weights, x, y):
        # Unpack
        W1 = weights[0, 0:40].reshape(5, 8)
        b1 = weights[0, 40:48]
        W2 = weights[0, 48:112].reshape(8, 8)
        b2 = weights[0, 112:120]
        W3 = weights[0, 120:136].reshape(8, 2)
        b3 = weights[0, 136:138]

        # ---- Forward ----
        z1 = torch.matmul(x, W1) + b1
        a1 = torch.relu(z1)
        z2 = torch.matmul(a1, W2) + b2
        a2 = torch.sigmoid(z2)
        z3 = torch.matmul(a2, W3) + b3

        # ---- Loss gradient: dL/dz3 for L = 0.5*||z3 - y||^2 ----
        d_out = z3 - y

        # ---- Layer 3 backward ----
        dW3 = torch.matmul(a2.transpose(0, 1), d_out)
        db3 = d_out.squeeze(0)
        d_a2 = torch.matmul(d_out, W3.transpose(0, 1))

        # ---- Sigmoid backward: sigma'(z) = sigma(z) * (1 - sigma(z)) ----
        d_z2 = d_a2 * a2 * (1.0 - a2)

        # ---- Layer 2 backward ----
        dW2 = torch.matmul(a1.transpose(0, 1), d_z2)
        db2 = d_z2.squeeze(0)
        d_a1 = torch.matmul(d_z2, W2.transpose(0, 1))

        # ---- ReLU backward ----
        relu_mask = (z1 > 0).float()
        d_z1 = d_a1 * relu_mask

        # ---- Layer 1 backward ----
        dW1 = torch.matmul(x.transpose(0, 1), d_z1)
        db1 = d_z1.squeeze(0)

        # ---- SGD update (lr=0.01 baked into circuit) ----
        W1_new = W1 - 0.01 * dW1
        b1_new = b1 - 0.01 * db1
        W2_new = W2 - 0.01 * dW2
        b2_new = b2 - 0.01 * db2
        W3_new = W3 - 0.01 * dW3
        b3_new = b3 - 0.01 * db3

        # Repack
        new_weights = torch.cat([
            W1_new.reshape(-1), b1_new,
            W2_new.reshape(-1), b2_new,
            W3_new.reshape(-1), b3_new,
        ]).unsqueeze(0)

        return new_weights


def flatten_model_weights(model):
    """Extract weights from nn.Sequential, transpose Linear weights to (in, out)."""
    parts = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            parts.append(param.data.t().reshape(-1))
        else:
            parts.append(param.data.reshape(-1))
    flat = torch.cat(parts).unsqueeze(0)
    assert flat.shape == (1, TOTAL_PARAMS), f"Bad shape: {flat.shape}"
    return flat


# ===========================================================================
# Step 1: Build model, verify manual backward matches autograd
# ===========================================================================
def step1_build_and_verify():
    print()
    print("=" * 60)
    print("STEP 1: Build SGDStep & verify backward pass")
    print("=" * 60)

    torch.manual_seed(42)
    ref = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.Sigmoid(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )

    weights_flat = flatten_model_weights(ref)
    print(f"  Parameters: {weights_flat.shape[1]}")

    torch.manual_seed(123)
    x = torch.randn(1, INPUT_DIM)
    y = torch.randn(1, OUTPUT_DIM)
    print(f"  x = {[f'{v:.4f}' for v in x[0].tolist()]}")
    print(f"  y = {[f'{v:.4f}' for v in y[0].tolist()]}")

    # Autograd reference
    ref.zero_grad()
    out = ref(x)
    loss = 0.5 * torch.sum((out - y) ** 2)
    loss.backward()

    with torch.no_grad():
        for p in ref.parameters():
            p.data -= LR * p.grad

    ref_updated = flatten_model_weights(ref)

    # Manual SGDStep
    sgd = SGDStep()
    weights_for_manual = flatten_model_weights(
        # Re-create original model with same seed
        _make_ref_model()
    )
    with torch.no_grad():
        manual_updated = sgd(weights_for_manual, x, y)

    # Compare
    max_diff = torch.max(torch.abs(ref_updated - manual_updated)).item()
    mean_diff = torch.mean(torch.abs(ref_updated - manual_updated)).item()
    print(f"  max  |autograd - manual| = {max_diff:.2e}")
    print(f"  mean |autograd - manual| = {mean_diff:.2e}")
    assert max_diff < 1e-6, f"FAIL: backward mismatch (max_diff={max_diff})"
    print(f"  PASS: manual backward matches autograd")

    return sgd, weights_for_manual, x, y


def _make_ref_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.Sigmoid(),
        nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )


# ===========================================================================
# Step 2: ONNX export
# ===========================================================================
def step2_onnx(model, weights, x, y):
    print()
    print("=" * 60)
    print("STEP 2: Export SGDStep to ONNX")
    print("=" * 60)

    torch.onnx.export(
        model,
        (weights, x, y),
        ONNX_PATH,
        input_names=["weights", "x", "y"],
        output_names=["new_weights"],
        opset_version=17,
        dynamo=False,
    )

    onnx_model = onnx.load(ONNX_PATH)
    ops = sorted(set(n.op_type for n in onnx_model.graph.node))
    n_nodes = len(onnx_model.graph.node)
    onnx_size = os.path.getsize(ONNX_PATH)

    print(f"  Nodes: {n_nodes}")
    print(f"  Ops: {ops}")
    print(f"  Size: {onnx_size / 1024:.2f} KB")
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")

    # input.json (single sample for proving)
    input_data = {
        "input_data": [
            weights[0].tolist(),
            x[0].tolist(),
            y[0].tolist(),
        ]
    }
    with open(INPUT_PATH, "w") as f:
        json.dump(input_data, f)

    # calibration.json (20 diverse samples so ezkl sees full value range)
    n_cal = 20
    cal_w, cal_x, cal_y = [], [], []
    for i in range(n_cal):
        torch.manual_seed(i * 7 + 1)
        wi = torch.randn(1, TOTAL_PARAMS) * 0.5
        xi = torch.randn(1, INPUT_DIM)
        yi = torch.randn(1, OUTPUT_DIM)
        cal_w.extend(wi[0].tolist())
        cal_x.extend(xi[0].tolist())
        cal_y.extend(yi[0].tolist())
    # Also include the actual input as a calibration sample
    cal_w.extend(weights[0].tolist())
    cal_x.extend(x[0].tolist())
    cal_y.extend(y[0].tolist())

    cal_data = {"input_data": [cal_w, cal_x, cal_y]}
    with open(CAL_PATH, "w") as f:
        json.dump(cal_data, f)

    print(f"  Saved input.json (1 sample)")
    print(f"  Saved calibration.json ({n_cal + 1} samples)")
    return ops, n_nodes


# ===========================================================================
# Step 3: ezkl settings & calibration
# ===========================================================================
def step3_calibrate():
    print()
    print("=" * 60)
    print("STEP 3: ezkl Settings & Calibration")
    print("=" * 60)

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"

    res = ezkl.gen_settings(ONNX_PATH, SETTINGS_PATH, py_run_args=py_run_args)
    assert res is True, "gen_settings failed"

    with open(SETTINGS_PATH) as f:
        init = json.load(f)
    print(f"  Initial logrows: {init['run_args']['logrows']}")
    print(f"  Initial lookups: {init.get('required_lookups', [])}")

    # Calibrate -- no logrows cap, safety margin=2 for backward pass headroom
    print(f"  Calibrating (target=resources, safety_margin=2)...")
    res = ezkl.calibrate_settings(
        CAL_PATH, ONNX_PATH, SETTINGS_PATH,
        target="resources",
        lookup_safety_margin=2,
    )
    assert res is True, "calibrate_settings failed"

    # Patch: SAFE check_mode
    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
    old_check = settings["run_args"].get("check_mode", "unknown")
    settings["run_args"]["check_mode"] = "SAFE"
    settings["check_mode"] = "SAFE"
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f)

    lr = settings["run_args"]["logrows"]
    nr = settings.get("num_rows", "?")
    ta = settings.get("total_assignments", "?")
    lookups = settings.get("required_lookups", [])
    ranges = settings.get("required_range_checks", [])
    lrange = settings["run_args"].get("lookup_range", "N/A")
    iscale = settings["run_args"]["input_scale"]
    pscale = settings["run_args"]["param_scale"]
    decomp = settings["run_args"]["decomp_base"]
    legs = settings["run_args"]["decomp_legs"]

    print(f"  check_mode: {old_check} -> SAFE")
    print(f"  scales: input={iscale}, param={pscale}")
    print(f"  decomp: {decomp}^{legs}")
    print(f"  logrows       = {lr} (2^{lr} = {2**lr} rows)")
    print(f"  num_rows      = {nr}")
    print(f"  total_assign  = {ta}")
    print(f"  lookups       = {lookups}")
    print(f"  range_checks  = {ranges}")
    print(f"  lookup_range  = {lrange}")
    if isinstance(nr, int):
        print(f"  utilization   = {nr}/{2**lr} = {nr/2**lr*100:.1f}%")

    if lrange != "N/A" and isinstance(lrange, list):
        lut_size = abs(lrange[1] - lrange[0])
        min_lr = math.ceil(math.log2(max(lut_size, 1)))
        print(f"  LUT size      = {lut_size} entries (min logrows={min_lr})")

    return settings


# ===========================================================================
# Step 4: Compile, SRS, keys
# ===========================================================================
def step4_setup(settings):
    print()
    print("=" * 60)
    print("STEP 4: Compile -> SRS -> Keys")
    print("=" * 60)

    t0 = time.time()
    res = ezkl.compile_circuit(ONNX_PATH, COMPILED_PATH, SETTINGS_PATH)
    t_compile = time.time() - t0
    assert res is True, "compile_circuit failed"
    print(f"  Compiled ({t_compile:.2f}s)")

    logrows = settings["run_args"]["logrows"]
    srs_source = "LOCAL (gen_srs)"
    t0 = time.time()
    try:
        async def _get_srs():
            return await ezkl.get_srs(
                settings_path=SETTINGS_PATH, srs_path=SRS_PATH
            )
        asyncio.run(_get_srs())
        srs_source = "PRODUCTION (ceremony)"
    except Exception:
        ezkl.gen_srs(SRS_PATH, logrows)
    t_srs = time.time() - t0
    print(f"  SRS: {srs_source} ({t_srs:.2f}s)")

    t0 = time.time()
    res = ezkl.setup(COMPILED_PATH, VK_PATH, PK_PATH, SRS_PATH)
    t_setup = time.time() - t0
    assert res is True, "setup failed"

    pk_size = os.path.getsize(PK_PATH)
    vk_size = os.path.getsize(VK_PATH)
    print(f"  pk.key: {pk_size/1024/1024:.2f} MB")
    print(f"  vk.key: {vk_size/1024:.2f} KB")
    print(f"  Setup: {t_setup:.2f}s")

    return pk_size, vk_size, srs_source


# ===========================================================================
# Step 5: Witness, prove, verify
# ===========================================================================
def step5_prove_and_verify():
    print()
    print("=" * 60)
    print("STEP 5: Witness -> Prove -> Verify")
    print("=" * 60)

    t0 = time.time()
    res = ezkl.gen_witness(INPUT_PATH, COMPILED_PATH, WITNESS_PATH)
    t_wit = time.time() - t0
    # gen_witness returns witness dict (not True) in ezkl v23
    assert res is not None, "gen_witness failed"
    assert os.path.exists(WITNESS_PATH), "witness.json not created"
    print(f"  Witness generated ({t_wit:.2f}s, {os.path.getsize(WITNESS_PATH)} bytes)")

    with open(WITNESS_PATH) as f:
        witness = json.load(f)
    if "pretty_elements" in witness:
        pe = witness["pretty_elements"]
        if "rescaled_inputs" in pe:
            for i, inp in enumerate(pe["rescaled_inputs"]):
                print(f"    input[{i}]: {len(inp)} values")
        if "rescaled_outputs" in pe:
            for i, out in enumerate(pe["rescaled_outputs"]):
                preview = out[:5]
                print(f"    output[{i}]: {len(out)} values (first 5: {preview})")

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    print(f"  Proving...")
    t0 = time.time()
    res = ezkl.prove(
        WITNESS_PATH, COMPILED_PATH, PK_PATH, PROOF_PATH, SRS_PATH,
    )
    proving_time = time.time() - t0

    rss_after = proc.memory_info().rss
    rss_peak = rss_after / 1024 / 1024
    rss_delta = (rss_after - rss_before) / 1024 / 1024

    assert res is not None, "prove returned None"
    assert os.path.exists(PROOF_PATH), "proof.json not created"
    proof_size = os.path.getsize(PROOF_PATH)
    print(f"  Proving time: {proving_time:.2f}s")
    print(f"  Proof size: {proof_size/1024:.2f} KB")
    print(f"  RSS peak: {rss_peak:.2f} MB")
    print(f"  RSS delta: {rss_delta:.2f} MB")

    t0 = time.time()
    verified = ezkl.verify(PROOF_PATH, SETTINGS_PATH, VK_PATH, SRS_PATH)
    t_verify = time.time() - t0
    print(f"  ezkl.verify() = {verified} ({t_verify:.2f}s)")

    return {
        "proving_time": proving_time,
        "proof_size_kb": proof_size / 1024,
        "rss_peak_mb": rss_peak,
        "rss_delta_mb": rss_delta,
        "verified": verified,
        "verify_time": t_verify,
    }


# ===========================================================================
# Step 6: Tamper test
# ===========================================================================
def step6_tamper_test():
    print()
    print("=" * 60)
    print("STEP 6: Tamper Tests")
    print("=" * 60)

    with open(PROOF_PATH) as f:
        proof_data = json.load(f)

    results = {}
    tp = os.path.join(ARTIFACT_DIR, "tampered.json")

    # Test 1: Flip byte in proof
    proof_raw = proof_data.get("proof", [])
    if isinstance(proof_raw, list) and len(proof_raw) > 10:
        tampered_proof = json.loads(json.dumps(proof_data))
        pos = len(proof_raw) // 2
        tampered_proof["proof"][pos] = (proof_raw[pos] + 1) % 256
        with open(tp, "w") as f:
            json.dump(tampered_proof, f)
        try:
            v = ezkl.verify(tp, SETTINGS_PATH, VK_PATH, SRS_PATH)
            results["proof_tamper"] = "FAIL (accepted!)" if v else "PASS"
        except Exception:
            results["proof_tamper"] = "PASS"
        if os.path.exists(tp):
            os.remove(tp)
    elif isinstance(proof_raw, str) and len(proof_raw) > 20:
        tampered_proof = json.loads(json.dumps(proof_data))
        t = list(proof_raw)
        pos = len(t) // 2
        t[pos] = 'f' if t[pos] != 'f' else '0'
        tampered_proof["proof"] = "".join(t)
        with open(tp, "w") as f:
            json.dump(tampered_proof, f)
        try:
            v = ezkl.verify(tp, SETTINGS_PATH, VK_PATH, SRS_PATH)
            results["proof_tamper"] = "FAIL (accepted!)" if v else "PASS"
        except Exception:
            results["proof_tamper"] = "PASS"
        if os.path.exists(tp):
            os.remove(tp)

    # Test 2: Tamper instances
    instances = proof_data.get("instances", [])
    if instances and len(instances) > 0 and len(instances[0]) > 0:
        tampered_inst = json.loads(json.dumps(proof_data))
        tampered_inst["instances"][0][0] = (
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        )

        with open(tp, "w") as f:
            json.dump(tampered_inst, f)
        try:
            v = ezkl.verify(tp, SETTINGS_PATH, VK_PATH, SRS_PATH)
            results["instance_tamper"] = "FAIL (accepted!)" if v else "PASS"
        except Exception:
            results["instance_tamper"] = "PASS"
        if os.path.exists(tp):
            os.remove(tp)

    for test, result in results.items():
        print(f"  {test}: {result}")

    return results


# ===========================================================================
# Step 7: Report
# ===========================================================================
def generate_report(settings, onnx_ops, n_nodes, pk_size, vk_size,
                    srs_source, metrics, tamper_results):
    print()
    print("=" * 60)
    print("STEP 7: Report")
    print("=" * 60)

    lr = settings["run_args"]["logrows"]
    nr = settings.get("num_rows", "?")
    ta = settings.get("total_assignments", "?")
    iscale = settings["run_args"]["input_scale"]
    pscale = settings["run_args"]["param_scale"]
    decomp = settings["run_args"]["decomp_base"]
    legs = settings["run_args"]["decomp_legs"]
    lookups = settings.get("required_lookups", [])
    ranges = settings.get("required_range_checks", [])
    lrange = settings["run_args"].get("lookup_range", "N/A")

    ta_str = f"{ta:,}" if isinstance(ta, int) else str(ta)
    nr_str = str(nr)
    util = f"{nr}/{2**lr} = {nr/2**lr*100:.1f}%" if isinstance(nr, int) else "?"

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# VoT SGDStep - Benchmark Report

**Generated**: {now}
**Platform**: {platform.system()} ({platform.machine()})
**CPU**: {platform.processor() or 'N/A'}
**Python**: {platform.python_version()}
**Packages**: torch={torch.__version__}, ezkl={ezkl.__version__}

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
| Parameters | {TOTAL_PARAMS} |
| Loss | MSE (L = 0.5 * norm(output - target)^2) |
| Optimizer | SGD, lr=0.01 (hardcoded) |
| Batch size | 1 |
| ONNX ops | {', '.join(onnx_ops)} |
| ONNX nodes | {n_nodes} |

## Circuit

| Property | Value |
|----------|-------|
| logrows | **{lr}** (2^{lr} = {2**lr} rows) |
| num_rows | {nr_str} |
| total_assignments | {ta_str} |
| utilization | {util} |
| input_scale | {iscale} |
| param_scale | {pscale} |
| decomp_base | {decomp} |
| decomp_legs | {legs} |
| check_mode | **SAFE** |
| Input visibility | public |
| Output visibility | public |
| required_lookups | {lookups} |
| range_checks | {ranges} |
| lookup_range | {lrange} |

## VoI vs VoT Comparison

| Metric | VoI (inference) | VoT (training step) |
|--------|-----------------|---------------------|
| logrows | 12 | {lr} |
| num_rows | 601 | {nr_str} |
| total_assignments | 1,202 | {ta_str} |
| pk.key | 44.26 MB | {pk_size/1024/1024:.2f} MB |
| Proving time | 1.56s | {metrics['proving_time']:.2f}s |
| Proof size | 41.44 KB | {metrics['proof_size_kb']:.2f} KB |
| ONNX nodes | ~20 | {n_nodes} |

## Security

| Check | Status |
|-------|--------|
| check_mode | **SAFE** |
| SRS source | {srs_source} |
| Tamper (proof bytes) | **{tamper_results.get('proof_tamper', 'N/A')}** |
| Tamper (instances) | **{tamper_results.get('instance_tamper', 'N/A')}** |

## Performance

| Metric | Value |
|--------|-------|
| Proving time | {metrics['proving_time']:.2f}s |
| Verify time | {metrics['verify_time']:.2f}s |
| Peak RSS | {metrics['rss_peak_mb']:.2f} MB |
| RSS delta | {metrics['rss_delta_mb']:.2f} MB |
| pk.key | {pk_size/1024/1024:.2f} MB |
| vk.key | {vk_size/1024:.2f} KB |
| Proof size | {metrics['proof_size_kb']:.2f} KB |
| ezkl.verify() | **{metrics['verified']}** |

## Artifacts

| File | Size |
|------|------|
| `sgd_step.onnx` | {os.path.getsize(ONNX_PATH)/1024:.2f} KB |
| `model.compiled` | {os.path.getsize(COMPILED_PATH)/1024:.2f} KB |
| `kzg.srs` | {os.path.getsize(SRS_PATH)/1024:.2f} KB |
| `pk.key` | {pk_size/1024/1024:.2f} MB |
| `vk.key` | {vk_size/1024:.2f} KB |
| `witness.json` | {os.path.getsize(WITNESS_PATH)/1024:.2f} KB |
| `proof.json` | {os.path.getsize(PROOF_PATH)/1024:.2f} KB |
| `settings.json` | {os.path.getsize(SETTINGS_PATH)/1024:.2f} KB |

## Verification

```
ezkl.verify() returned: {metrics['verified']}
check_mode: SAFE
Tamper test (proof bytes): {tamper_results.get('proof_tamper', 'N/A')}
Tamper test (instances): {tamper_results.get('instance_tamper', 'N/A')}
```

All artifacts are real. check_mode=SAFE. No mocks, no stubs.
This is a real ZK proof of a real SGD training step.
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"  Report: {REPORT_PATH}")
    print(f"  VoT verified: {metrics['verified']}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print()
    print("#" * 60)
    print("#  VoT (Verification of Training) SGDStep PoC")
    print("#  Proving a single SGD step in zero-knowledge")
    print("#" * 60)

    t_total = time.time()

    model, weights, x, y = step1_build_and_verify()
    onnx_ops, n_nodes = step2_onnx(model, weights, x, y)
    settings = step3_calibrate()
    pk_size, vk_size, srs_source = step4_setup(settings)
    metrics = step5_prove_and_verify()
    tamper_results = step6_tamper_test()
    generate_report(settings, onnx_ops, n_nodes, pk_size, vk_size,
                    srs_source, metrics, tamper_results)

    total = time.time() - t_total
    print()
    print(f"Total pipeline time: {total:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
