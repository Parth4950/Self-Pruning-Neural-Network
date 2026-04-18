# Self-Pruning Neural Network

## Overview

This project implements a self-pruning feedforward neural network in PyTorch for CIFAR-10 image classification.
Instead of pruning weights after training, the model learns which weights to suppress during training using differentiable gates.

Each weight has a learnable gate, and training optimizes both:
- classification performance
- model sparsity

This makes pruning part of the optimization process, not a separate post-processing step.

## Key Idea

The core idea is a custom layer called `PrunableLinear`:

- Standard learnable weight: `W`
- Learnable gate scores: `S` (same shape as `W`)
- Gates: `G = sigmoid(S)` (values in `(0, 1)`)
- Effective weight used in forward pass: `W_pruned = W * G`

So each weight is softly controlled by its gate:
- gate near `1` -> weight remains active
- gate near `0` -> weight is effectively pruned

Because gates come from `sigmoid`, everything remains differentiable and trainable with backpropagation.

## Model Architecture

The model (`PrunableNet`) is a simple MLP for CIFAR-10:

1. Input image: `(3, 32, 32)`
2. Flatten to `3072`
3. `PrunableLinear(3072, 512)` + ReLU
4. `PrunableLinear(512, 256)` + ReLU
5. `PrunableLinear(256, 10)` -> logits

No softmax is applied in the model head (required for `CrossEntropyLoss`).

## Loss Function

Training uses a combined objective:

`Total Loss = CrossEntropyLoss + lambda * Sparsity Loss`

Where:
- CrossEntropyLoss drives classification accuracy
- Sparsity Loss encourages gate values to shrink

Sparsity loss is computed as the L1-style sum over gates:

`Sparsity Loss = sum(G)`

## Why L1 Encourages Sparsity

L1-style penalties encourage many values to become small (often near zero) rather than distributing shrinkage evenly.

In this project:
- penalizing the sum of gate activations pushes unnecessary gates downward
- lower gates suppress corresponding weights
- this naturally increases sparsity while preserving useful weights

Intuition: the model pays a cost for keeping gates open, so it learns to keep only the important connections active.

## Experiments

Experiments were run with:

- `lambda = [1e-3, 1e-2, 1e-1]`
- CIFAR-10 training with fixed random seed for reproducibility
- Metrics tracked per lambda:
  - Test Accuracy
  - Overall Sparsity (% gates below threshold)
  - Layer-wise sparsity

## Results Table

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|--------------------|--------------|
| 0.001   | 53.26%              | 0.04%        |
| 0.01   | 52.55%              | 0.07%        |
| 0.1   | 53.77%              | 6.57%        |

## Observations

- Increasing `lambda` strengthens sparsity pressure.
- Higher sparsity usually comes with some accuracy drop.
- Lower `lambda` preserves accuracy better but prunes less.
- The project highlights the expected tradeoff between compactness and performance.

## Visualizations

The training script generates and/or reports:

1. Lambda vs Accuracy (log-scale lambda axis)
2. Lambda vs Sparsity (log-scale lambda axis)
3. Gate value distribution (to inspect how gates shift toward low values)

It also logs:
- `results.csv` with `lambda, accuracy, sparsity`
- layer-wise sparsity percentages for each experiment

## Conclusion

This project demonstrates a practical way to build train-time sparse neural networks using learnable gates.
By combining classification and sparsity objectives, the model can discover a balance between accuracy and parameter efficiency in a single training loop.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- CSV (standard library)

## How to Run

```bash
# 1) Install dependencies
pip install torch torchvision numpy matplotlib

# 2) Run training
python main.py
```

Expected outputs:
- console logs for loss/accuracy/sparsity/layer-wise sparsity
- `results.csv`
- plots for lambda-accuracy and lambda-sparsity (and gate distribution if enabled)

## Key Takeaways

- Differentiable gating is a clean way to integrate pruning into training.
- L1-style regularization on gates is effective for inducing sparsity.
- `lambda` acts as a direct control knob for the sparsity-accuracy tradeoff.
- Simple architectures can clearly demonstrate pruning behavior and analysis workflows.
