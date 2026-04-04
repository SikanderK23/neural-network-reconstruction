import torch
import glob
import json
import os

print("Running converter...")

paths = sorted(glob.glob("pieces/piece_*.pth"))

print(f"Found {len(paths)} .pth files")

if len(paths) == 0:
    print("ERROR: No .pth files found in pieces/")
    exit()

pieces = {}

for path in paths:
    idx = int(path.split('_')[1].split('.')[0])
    data = torch.load(path, map_location='cpu')

    pieces[idx] = {
        "weight": data["weight"].tolist(),
        "bias": data["bias"].tolist()
    }

with open("pieces_data.json", "w") as f:
    json.dump(pieces, f)

print("Saved pieces_data.json")
print("File exists:", os.path.exists("pieces_data.json"))