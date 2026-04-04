# Neural Network Reconstruction

This project reconstructs the correct ordering of a scrambled residual neural network using optimization techniques.

## Problem

A trained neural network was broken into individual linear layers.  
The goal is to recover the correct ordering of these layers using only:
- layer weights
- historical input/output data
- knowledge of the residual block structure

## Approach

The solution uses a multi-stage optimization pipeline:

1. **Layer Pairing**
   - Hungarian algorithm to match input/output layers

2. **Initial Ordering**
   - Beam search to construct a strong starting sequence

3. **Refinement**
   - Simulated annealing to minimize prediction error

## Technologies

- Python
- PyTorch
- NumPy
- SciPy

## Results

- Successfully reconstructs the model structure
- Achieves near-zero mean squared error (MSE)

## How to Run

1. Place the required files:
   - `historical_data.csv`
   - `pieces/` folder with `piece_*.pth`

2. Install dependencies:

```bash
pip3 install -r requirements.txt
