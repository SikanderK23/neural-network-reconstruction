"""
Jane Street - Dropped a Neural Net
Solver: Simulated Annealing (type-safe, residual block ordering)

The puzzle provides 97 weight pieces from a dismantled residual network.
96 of them form 48 (Type-A, Type-B) pairs that make up residual blocks,
and piece 85 is the fixed output layer. The goal is to find the correct
pairing and ordering of those 48 blocks that minimises MSE on the
provided historical data.

Approach:
  - Pieces are classified by weight shape:
      Type-A: (96, 48)  — expand 48 features to 96 (the "up" projection)
      Type-B: (48, 96)  — compress 96 features back to 48 (the "down" projection)
  - Each residual block applies: x = x + B(ReLU(A(x)))
  - Simulated annealing explores the space of (pairing, ordering) assignments
    via three move types, accepting worse states with a temperature-scaled
    probability to escape local minima.
  - Temperature is manually lowered across multiple runs, each time seeding
    from the best sequence found so far, until MSE converges to 0.

Hardware: Apple M1 Pro — uses MPS (Metal Performance Shaders) GPU backend.
"""

import torch
import torch.nn.functional as F
import glob
import csv
import time
import numpy as np
import random
import math

# ---------------------------------------------------------------------------
# Device selection
# Prefer Apple MPS on M1/M2, fall back to CUDA then CPU.
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# 1. Load historical data
#    The CSV contains 10,000 rows. The first 48 columns are input features (X)
#    and the 'pred' column is the target value we want the network to predict (T).
# ---------------------------------------------------------------------------
print("Loading data...")
with open("historical_data.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [list(map(float, r)) for r in reader]
data = np.array(rows, dtype=np.float32)
X = torch.tensor(data[:, :48], device=device)   # shape: (10000, 48)
T = torch.tensor(data[:, header.index('pred')], device=device)  # shape: (10000,)

# ---------------------------------------------------------------------------
# 2. Load weight pieces and classify by shape
#    Each piece_N.pth file contains a 'weight' matrix and a 'bias' vector.
#    Shape tells us the role:
#      (96, 48) → Type-A: expands the 48-dim input to a 96-dim hidden state
#      (48, 96) → Type-B: projects the 96-dim hidden state back down to 48-dim
#    Piece 85 is the output layer (1, 48) and is always placed last — not shuffled.
# ---------------------------------------------------------------------------
print("Loading pieces and identifying shapes...")
pw, pb = {}, {}      # pw[i] = weight tensor, pb[i] = bias tensor
type_a_pool, type_b_pool = [], []

for path in sorted(glob.glob("pieces/piece_*.pth"), key=lambda p: int(p.split('_')[1].split('.')[0])):
    idx = int(path.split('_')[1].split('.')[0])
    d = torch.load(path, map_location='cpu', weights_only=True)
    pw[idx] = d['weight'].to(device)
    pb[idx] = d['bias'].to(device)

    if idx == 85:
        continue  # output piece — fixed, not part of the search space

    if pw[idx].shape[1] == 48:
        type_a_pool.append(idx)   # expander pieces
    elif pw[idx].shape[1] == 96:
        type_b_pool.append(idx)   # compressor pieces

# Grab the output layer weights separately for use in scoring
w_last, b_last = pw[85], pb[85]
print(f"Found {len(type_a_pool)} Type-A pieces and {len(type_b_pool)} Type-B pieces.")

# ---------------------------------------------------------------------------
# 3. Core functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def score(pairs):
    """
    Run the full forward pass with the given block ordering and return MSE.

    `pairs` is a list of 48 (type_a_idx, type_b_idx) tuples defining both
    the pairing and the order of residual blocks.

    Each block computes:
        h = ReLU(A(x))        — expand to 96 dims, apply non-linearity
        x = x + B(h)          — compress back to 48 dims, add residual

    After all 48 blocks, the output layer (piece 85) maps 48 → 1 prediction.
    MSE is averaged over all 10,000 data rows.
    """
    x = X
    for a, b in pairs:
        h = F.linear(x, pw[a], pb[a])   # Type-A: (10000, 48) -> (10000, 96)
        torch.relu_(h)                   # in-place ReLU — zero out negatives
        x = x + F.linear(h, pw[b], pb[b])  # Type-B + residual skip connection
    return ((F.linear(x, w_last, b_last).squeeze(-1) - T)**2).mean().item()


def save_best(pairs, mse):
    """
    Persist the current best solution to disk as a flat index list.
    Format matches the puzzle's expected submission: [a0, b0, a1, b1, ..., 85]
    """
    flat = [v for a, b in pairs for v in [a, b]] + [85]
    with open("best_answer.txt", "w") as f:
        f.write(f"MSE: {mse}\n")
        f.write(str(flat) + "\n")
    print(f"  -> Saved best_answer.txt")

# ---------------------------------------------------------------------------
# 4. Initial state
#    Seeded from the best sequence found in the previous annealing round.
#    On each new run, copy the list from best_answer.txt into best_flat below,
#    then lower `temp` in section 5 to tighten the search.
# ---------------------------------------------------------------------------
best_flat = [43, 34, 65, 22, 69, 89, 28, 12, 27, 76, 81, 8,
             64, 70, 5, 21, 62, 79, 94, 96, 4, 17, 48, 9, 23,
             46, 95, 26, 14, 33, 1, 40, 50, 66, 15, 67, 16, 83,
             41, 92, 77, 32, 10, 20, 3, 53, 45, 19, 87, 71, 88, 54,
             39, 38, 18, 25, 56, 30, 91, 29, 35, 24, 44, 82, 61, 80,
             86, 57, 31, 36, 13, 7, 68, 47, 59, 52, 84, 63, 74, 90, 0,
             75, 73, 11, 37, 6, 58, 78, 42, 55, 49, 72, 2, 51, 60, 93,
             85]

# Convert flat list into (Type-A, Type-B) pairs for the score function
best_pairs = [(best_flat[i], best_flat[i+1]) for i in range(0, 96, 2)]
current_pairs = list(best_pairs)
best_mse = score(best_pairs)
current_mse = best_mse
print(f"Starting Type-Safe Annealing at MSE: {best_mse:.10f}")

# ---------------------------------------------------------------------------
# 5. Hyperparameters
#    temp         — starting temperature. Controls how often worse moves are
#                   accepted. Lowered manually between runs as MSE converges.
#    cooling_rate — multiplicative decay applied to temp each iteration.
#                   0.999998 over 2M iterations reduces temp by ~86%.
#    iterations   — total number of proposed moves per run.
# ---------------------------------------------------------------------------
temp = 0.00001
cooling_rate = 0.999998
iterations = 2_000_000

# ---------------------------------------------------------------------------
# 6. Simulated annealing loop
#
#    Each iteration proposes one of three type-safe mutations:
#      Move 1 (40%): swap the Type-A piece between two randomly chosen blocks
#      Move 2 (40%): swap the Type-B piece between two randomly chosen blocks
#      Move 3 (20%): swap two entire (A, B) blocks, changing their order
#
#    A move is accepted if it improves MSE, or with probability
#    exp(-delta / temp) if it doesn't — allowing occasional uphill moves
#    to escape local minima. As temp cools, only improvements are accepted.
# ---------------------------------------------------------------------------
start_time = time.time()
try:
    for i in range(iterations):
        new_pairs = [list(p) for p in current_pairs]
        move_type = random.random()

        if move_type < 0.4:
            # Swap the Type-A (expander) piece between two blocks
            # Changes what each block does in the expand step, not the order
            idx1, idx2 = random.sample(range(len(new_pairs)), 2)
            new_pairs[idx1][0], new_pairs[idx2][0] = new_pairs[idx2][0], new_pairs[idx1][0]

        elif move_type < 0.8:
            # Swap the Type-B (compressor) piece between two blocks
            # Changes what each block does in the compress+residual step
            idx1, idx2 = random.sample(range(len(new_pairs)), 2)
            new_pairs[idx1][1], new_pairs[idx2][1] = new_pairs[idx2][1], new_pairs[idx1][1]

        else:
            # Swap two entire blocks — changes the order of transformations
            # in the forward pass without altering any pairings
            idx1, idx2 = random.sample(range(len(new_pairs)), 2)
            new_pairs[idx1], new_pairs[idx2] = new_pairs[idx2], new_pairs[idx1]

        final_new_pairs = [tuple(p) for p in new_pairs]
        new_mse = score(final_new_pairs)
        delta = new_mse - current_mse  # positive = worse, negative = better

        # Accept the move if it improves MSE, or probabilistically if it doesn't
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            current_pairs = final_new_pairs
            current_mse = new_mse

            # Track and save the all-time best
            if current_mse < best_mse - 1e-9:
                best_mse = current_mse
                best_pairs = list(current_pairs)
                save_best(best_pairs, best_mse)
                print(f"Iter {i} | New Best: {best_mse:.10f} | Temp: {temp:.8f}")

        temp *= cooling_rate  # cool down after every iteration

        if i % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{iterations} | Current MSE: {current_mse:.8f} | Best MSE: {best_mse:.8f} | Temp: {temp:.8f} | {elapsed:.0f}s")

except KeyboardInterrupt:
    print("\nStopped early. Saving current best...")

print(f"\nFinal Best MSE: {best_mse:.10f}")
save_best(best_pairs, best_mse)
