import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable weight pruning.

    Each weight has a corresponding learnable gate score. The gate score is
    passed through a sigmoid to produce a soft gate value in [0, 1], which
    smoothly controls whether a weight is active (near 1) or pruned (near 0).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix: shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Learnable bias vector: shape (out_features,)
        self.bias = nn.Parameter(torch.empty(out_features))

        # Learnable gate scores, one per weight: same shape as weight
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Kaiming uniform for stable training
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize bias similarly to nn.Linear defaults
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Start gate scores near zero so sigmoid outputs ~0.5 initially
        # (allows gradients to flow well and avoids hard pruning at init)
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

    def _compute_pruned_weight(self):
        # Sigmoid maps gate scores to (0, 1), giving differentiable soft gates.
        # This keeps pruning learnable via gradient descent.
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise gating prunes/suppresses weights continuously.
        pruned_weight = self.weight * gates
        return pruned_weight

    def forward(self, x):
        pruned_weight = self._compute_pruned_weight()
        return F.linear(x, pruned_weight, self.bias)


# -----------------------
# Small test snippet
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    in_features = 10
    out_features = 6

    x = torch.randn(batch_size, in_features)
    layer = PrunableLinear(in_features, out_features)

    y = layer(x)
    print("Output shape:", y.shape)