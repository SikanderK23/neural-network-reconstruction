#!/usr/bin/env python3
"""
Solve the residual network ordering puzzle — GPU-accelerated (PyTorch + CUDA).

Run:
    conda activate pytorch
    python solve_ordering_gpu.py

Strategy:
  1. Score all 48×48 (inp, out) pairings on GPU, find optimal via Hungarian.
  2. Beam-search greedy forward construction (all matmuls on GPU).
  3. Simulated annealing on block ordering (forward pass on GPU per iteration).
  4. Joint pairing + ordering annealing if MSE still > 0.01.
"""

import json
import csv
import time
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# ============================================================
# CONFIG
# ============================================================
BEAM_WIDTH        = 50
ANNEAL_ITERATIONS = 5_000_000
ANNEAL_T_START    = 0.05
ANNEAL_T_END      = 1e-7
JOINT_ITERS       = 5_000_000
SEED              = 42
DATA_FILE         = "historical_data.csv"
PIECES_FILE       = "pieces_data.json"

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")
with open(PIECES_FILE) as f:
    pieces_raw = json.load(f)

with open(DATA_FILE) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [list(map(float, r)) for r in reader]

data_np = np.array(rows, dtype=np.float32)

X_all   = torch.tensor(data_np[:, :48],  device=device)   # (10000, 48)
TRUE_all = torch.tensor(data_np[:, -1],  device=device)   # (10000,)

# Fast subset for annealing
torch.manual_seed(SEED)
np.random.seed(SEED)
subset_idx = np.random.choice(len(X_all), size=2000, replace=False)
X_fast    = X_all[subset_idx]
TRUE_fast = TRUE_all[subset_idx]

# ============================================================
# PIECE WEIGHTS → GPU
# ============================================================
piece_w = {}   # idx → (rows, cols) weight tensor on GPU
piece_b = {}   # idx → bias tensor on GPU

for k, v in pieces_raw.items():
    idx = int(k)
    piece_w[idx] = torch.tensor(v["weight"], dtype=torch.float32, device=device)
    piece_b[idx] = torch.tensor(v["bias"],   dtype=torch.float32, device=device)

w_last = piece_w[85]   # (1, 48)
b_last = piece_b[85]   # (1,)

# Classify pieces by shape
inp_pieces = sorted(i for i in range(97) if i != 85 and piece_w[i].shape == (96, 48))
out_pieces = sorted(i for i in range(97) if i != 85 and piece_w[i].shape == (48, 96))
print(f"Inp pieces: {len(inp_pieces)}, Out pieces: {len(out_pieces)}")

# ============================================================
# CORE GPU FUNCTIONS
# ============================================================
@torch.no_grad()
def apply_block(x: torch.Tensor, inp_idx: int, out_idx: int) -> torch.Tensor:
    """Apply one residual block: x + out(ReLU(inp(x)))"""
    h = torch.nn.functional.linear(x, piece_w[inp_idx], piece_b[inp_idx])
    h = torch.relu(h)
    return x + torch.nn.functional.linear(h, piece_w[out_idx], piece_b[out_idx])

@torch.no_grad()
def compute_mse(x: torch.Tensor, targets: torch.Tensor) -> float:
    """Apply last layer and compute MSE."""
    pred = torch.nn.functional.linear(x, w_last, b_last).squeeze(-1)
    return torch.mean((pred - targets) ** 2).item()

@torch.no_grad()
def full_forward(block_order, X: torch.Tensor, targets: torch.Tensor) -> float:
    """Run all blocks in order and return MSE."""
    x = X.clone()
    for inp_idx, out_idx in block_order:
        x = apply_block(x, inp_idx, out_idx)
    return compute_mse(x, targets)

# ============================================================
# STEP 1: PAIRINGS — score all 48×48 combos on GPU
# ============================================================
print("\n=== Step 1: Determine correct pairings ===")
X_sample = X_all[:500]                        # (500, 48)

pair_scores = torch.zeros(48, 48, device=device)
for ii, inp_idx in enumerate(inp_pieces):
    h = torch.nn.functional.linear(X_sample, piece_w[inp_idx], piece_b[inp_idx])
    h = torch.relu(h)                         # (500, 96)
    for oi, out_idx in enumerate(out_pieces):
        res = torch.nn.functional.linear(h, piece_w[out_idx], piece_b[out_idx])
        pair_scores[ii, oi] = res.abs().mean()

ps_cpu = pair_scores.cpu().numpy()
print(f"Pair score range: [{ps_cpu.min():.4f}, {ps_cpu.max():.4f}], mean: {ps_cpu.mean():.4f}")

# Confirmed pairs
confirmed_pairs = [
    (0,75),(1,40),(2,51),(3,30),(4,34),(5,83),(10,20),(13,7),(14,33),(15,67),
    (16,82),(18,25),(23,22),(27,76),(28,12),(31,36),(35,66),(37,19),(39,32),(41,57),
    (42,55),(43,8),(44,90),(45,6),(48,9),(49,72),(50,92),(56,54),(58,78),(59,52),
    (60,21),(61,80),(62,79),(64,70),(65,24),(68,47),(69,89),(73,11),(74,38),(77,53),
    (81,93),(84,46),(86,71),(87,63),(88,17),(91,29),(94,96),(95,26)
]
confirmed_set = set(confirmed_pairs)

# Hungarian — minimise residual magnitude
row_min, col_min = linear_sum_assignment(ps_cpu)
hungarian_min = [(inp_pieces[r], out_pieces[c]) for r, c in zip(row_min, col_min)]
diff_min = set(hungarian_min) - confirmed_set
print(f"\nHungarian (min) differs from confirmed: {len(diff_min)} pairs")
for p in sorted(diff_min):
    print(f"  {p}")

# Hungarian — maximise
row_max, col_max = linear_sum_assignment(-ps_cpu)
hungarian_max = [(inp_pieces[r], out_pieces[c]) for r, c in zip(row_max, col_max)]
diff_max = set(hungarian_max) - confirmed_set
print(f"Hungarian (max) differs from confirmed: {len(diff_max)} pairs")

# Alternative pairs (swap the 4 ambiguous from user's second run)
alt_pairs = [p for p in confirmed_pairs if p not in {(4,34),(43,8),(37,19),(50,92)}]
alt_pairs += [(43,34),(4,19),(50,8),(37,92)]

pairing_options = {
    "confirmed":     confirmed_pairs,
    "alt_swap":      alt_pairs,
    "hungarian_min": hungarian_min,
    "hungarian_max": hungarian_max,
}

# ============================================================
# STEP 2: BEAM SEARCH (GPU)
# ============================================================
print(f"\n=== Step 2: Beam Search (width={BEAM_WIDTH}) ===")

@torch.no_grad()
def beam_search(pairs, X, targets, beam_width=BEAM_WIDTH):
    n = len(pairs)
    initial_mse = compute_mse(X, targets)
    # Each beam entry: (mse, order_indices, x_state)
    beam = [(initial_mse, [], X.clone())]

    for step in range(n):
        candidates = []
        for mse_val, order, x in beam:
            used = set(order)
            for i in range(n):
                if i in used:
                    continue
                inp_idx, out_idx = pairs[i]
                x_new = apply_block(x, inp_idx, out_idx)
                m = compute_mse(x_new, targets)
                candidates.append((m, order + [i], x_new))

        candidates.sort(key=lambda c: c[0])
        beam = candidates[:beam_width]

        if step % 10 == 0 or step == n - 1:
            print(f"  Step {step:2d}: best MSE = {beam[0][0]:.10f}  "
                  f"(candidates evaluated: {len(candidates)})")
    return beam[0]

best_global_mse   = float("inf")
best_global_order  = None
best_global_pairs  = None

for name, pairs in pairing_options.items():
    print(f"\n--- Pairing: {name} ---")
    t0 = time.time()
    mse_val, order_idx, _ = beam_search(pairs, X_fast, TRUE_fast, BEAM_WIDTH)
    elapsed = time.time() - t0

    block_order = [pairs[i] for i in order_idx]
    full_mse = full_forward(block_order, X_all, TRUE_all)
    print(f"  Beam MSE (subset): {mse_val:.10f}")
    print(f"  Full MSE:          {full_mse:.10f}   ({elapsed:.1f}s)")

    if full_mse < best_global_mse:
        best_global_mse   = full_mse
        best_global_order  = order_idx
        best_global_pairs  = pairs

print(f"\nBest after beam search: MSE = {best_global_mse:.10f}")

# ============================================================
# STEP 3: SIMULATED ANNEALING on ordering (GPU eval)
# ============================================================
print(f"\n=== Step 3: Simulated Annealing ({ANNEAL_ITERATIONS:,} iters) ===")

pairs    = best_global_pairs
order    = list(best_global_order)
n        = len(order)
rng      = np.random.RandomState(SEED)

@torch.no_grad()
def eval_order(order, pairs, X, targets):
    x = X.clone()
    for idx in order:
        inp_i, out_i = pairs[idx]
        x = apply_block(x, inp_i, out_i)
    return compute_mse(x, targets)

current_mse = eval_order(order, pairs, X_fast, TRUE_fast)
best_mse    = current_mse
best_order  = list(order)

t0 = time.time()
accepted = improved = 0

for it in range(ANNEAL_ITERATIONS):
    progress = it / ANNEAL_ITERATIONS
    T = ANNEAL_T_START * (ANNEAL_T_END / ANNEAL_T_START) ** progress

    move = rng.randint(5)

    # ---- apply move (in-place) ----
    if move == 0:                           # random swap
        i, j = rng.choice(n, 2, replace=False)
        order[i], order[j] = order[j], order[i]
    elif move == 1:                         # adjacent swap
        i = rng.randint(n - 1)
        j = i + 1
        order[i], order[j] = order[j], order[i]
    elif move == 2:                         # insert
        i = rng.randint(n)
        j = rng.randint(n)
        if i != j:
            blk = order.pop(i)
            order.insert(j, blk)
    elif move == 3:                         # reverse segment
        i, j = sorted(rng.choice(n, 2, replace=False))
        if j - i <= 12:
            order[i:j+1] = order[i:j+1][::-1]
    else:                                   # rotate segment
        i, j = sorted(rng.choice(n, 2, replace=False))
        if j - i <= 8:
            seg = order[i:j+1]
            order[i:j+1] = seg[1:] + seg[:1]

    new_mse = eval_order(order, pairs, X_fast, TRUE_fast)
    delta   = new_mse - current_mse

    if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-30)):
        current_mse = new_mse
        accepted += 1
        if new_mse < best_mse:
            best_mse  = new_mse
            best_order = list(order)
            improved += 1
    else:
        # ---- undo move ----
        if move == 0:
            order[i], order[j] = order[j], order[i]
        elif move == 1:
            order[i], order[j] = order[j], order[i]
        elif move == 2:
            if i != j:
                blk = order.pop(j)
                order.insert(i, blk)
        elif move == 3:
            if j - i <= 12:
                order[i:j+1] = order[i:j+1][::-1]
        else:
            if j - i <= 8:
                seg = order[i:j+1]
                order[i:j+1] = seg[-1:] + seg[:-1]

    if it % 500_000 == 0:
        elapsed = time.time() - t0
        full_chk = eval_order(best_order, pairs, X_all, TRUE_all)
        print(f"  {it:>9,} | T={T:.2e} | cur={current_mse:.8f} "
              f"| best(sub)={best_mse:.8f} | best(full)={full_chk:.8f} "
              f"| acc={accepted} imp={improved} | {elapsed:.0f}s")

order = best_order

# ============================================================
# STEP 4: Final results (ordering-only)
# ============================================================
block_order    = [pairs[i] for i in order]
final_mse_full = full_forward(block_order, X_all, TRUE_all)

flat_order = []
for inp_i, out_i in block_order:
    flat_order += [inp_i, out_i]
flat_order.append(85)

print(f"\n=== After ordering annealing ===")
print(f"MSE (full): {final_mse_full:.10f}")
print(f"Order:      {flat_order}")

# ============================================================
# STEP 5: JOINT pairing + ordering annealing (if still > 0.01)
# ============================================================
if final_mse_full > 0.01:
    print(f"\n=== Step 4: Joint pairing+ordering annealing ({JOINT_ITERS:,} iters) ===")

    inp_to_ii = {v: i for i, v in enumerate(inp_pieces)}
    out_to_oi = {v: i for i, v in enumerate(out_pieces)}

    pairing_perm   = [0] * 48
    block_ordering = []
    for inp_i, out_i in block_order:
        ii = inp_to_ii[inp_i]
        oi = out_to_oi[out_i]
        pairing_perm[ii] = oi
        block_ordering.append(ii)

    @torch.no_grad()
    def eval_joint(pp, bo, X, targets):
        x = X.clone()
        for bi in bo:
            x = apply_block(x, inp_pieces[bi], out_pieces[pp[bi]])
        return compute_mse(x, targets)

    cur_mse     = eval_joint(pairing_perm, block_ordering, X_fast, TRUE_fast)
    best_j_mse  = cur_mse
    best_pp     = list(pairing_perm)
    best_bo     = list(block_ordering)

    t0 = time.time()
    accepted = improved = 0

    for it in range(JOINT_ITERS):
        progress = it / JOINT_ITERS
        T = 0.1 * (1e-7 / 0.1) ** progress

        move = rng.randint(7)

        if move <= 1:                           # swap pairing
            i, j = rng.choice(48, 2, replace=False)
            pairing_perm[i], pairing_perm[j] = pairing_perm[j], pairing_perm[i]
        elif move == 2:                         # swap ordering
            i, j = rng.choice(48, 2, replace=False)
            block_ordering[i], block_ordering[j] = block_ordering[j], block_ordering[i]
        elif move == 3:                         # adjacent swap ordering
            i = rng.randint(47)
            block_ordering[i], block_ordering[i+1] = block_ordering[i+1], block_ordering[i]
        elif move == 4:                         # insert ordering
            i = rng.randint(48); j = rng.randint(48)
            if i != j:
                blk = block_ordering.pop(i)
                block_ordering.insert(j, blk)
        elif move == 5:                         # swap pair + ordering together
            i, j = rng.choice(48, 2, replace=False)
            pairing_perm[i], pairing_perm[j] = pairing_perm[j], pairing_perm[i]
            pi = block_ordering.index(i)
            pj = block_ordering.index(j)
            block_ordering[pi], block_ordering[pj] = block_ordering[pj], block_ordering[pi]
        else:                                   # reverse segment
            i, j = sorted(rng.choice(48, 2, replace=False))
            if j - i <= 10:
                block_ordering[i:j+1] = block_ordering[i:j+1][::-1]

        new_mse = eval_joint(pairing_perm, block_ordering, X_fast, TRUE_fast)
        delta   = new_mse - cur_mse

        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-30)):
            cur_mse = new_mse
            accepted += 1
            if new_mse < best_j_mse:
                best_j_mse = new_mse
                best_pp    = list(pairing_perm)
                best_bo    = list(block_ordering)
                improved  += 1
        else:
            # ---- undo ----
            if move <= 1:
                pairing_perm[i], pairing_perm[j] = pairing_perm[j], pairing_perm[i]
            elif move == 2:
                block_ordering[i], block_ordering[j] = block_ordering[j], block_ordering[i]
            elif move == 3:
                block_ordering[i], block_ordering[i+1] = block_ordering[i+1], block_ordering[i]
            elif move == 4:
                if i != j:
                    blk = block_ordering.pop(j)
                    block_ordering.insert(i, blk)
            elif move == 5:
                pairing_perm[i], pairing_perm[j] = pairing_perm[j], pairing_perm[i]
                block_ordering[pi], block_ordering[pj] = block_ordering[pj], block_ordering[pi]
            else:
                if j - i <= 10:
                    block_ordering[i:j+1] = block_ordering[i:j+1][::-1]

        if it % 500_000 == 0:
            elapsed = time.time() - t0
            full_chk = eval_joint(best_pp, best_bo, X_all, TRUE_all)
            print(f"  {it:>9,} | T={T:.2e} | cur={cur_mse:.8f} "
                  f"| best(sub)={best_j_mse:.8f} | best(full)={full_chk:.8f} "
                  f"| acc={accepted} imp={improved} | {elapsed:.0f}s")

    # Rebuild final answer from joint search
    final_bo_joint = []
    for bi in best_bo:
        final_bo_joint.append((inp_pieces[bi], out_pieces[best_pp[bi]]))

    joint_mse = full_forward(final_bo_joint, X_all, TRUE_all)
    print(f"\nJoint search MSE (full): {joint_mse:.10f}")

    if joint_mse < final_mse_full:
        flat_order = []
        for inp_i, out_i in final_bo_joint:
            flat_order += [inp_i, out_i]
        flat_order.append(85)
        final_mse_full = joint_mse
        print("*** Joint search improved the result! ***")

# ============================================================
# FINAL OUTPUT
# ============================================================
print(f"\n{'='*60}")
print(f"FINAL ANSWER")
print(f"  MSE   = {final_mse_full:.10f}")
print(f"  Order = {flat_order}")
print(f"{'='*60}")
