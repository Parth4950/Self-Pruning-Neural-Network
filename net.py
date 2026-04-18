import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PrunableLinear


class PrunableNet(nn.Module):
    """
    Feedforward CIFAR-10 classifier built with prunable linear layers.

    Each PrunableLinear layer uses learnable gates over weights, enabling
    dynamic, differentiable pruning behavior during training.
    """

    def __init__(self):
        super().__init__()

        # Fully connected prunable layers:
        # 3072 -> 512 -> 256 -> 10
        self.fc1 = PrunableLinear(3 * 32 * 32, 512) #input layer
        self.fc2 = PrunableLinear(512, 256) #hidden layer
        self.fc3 = PrunableLinear(256, 10) #output layer

    def forward(self, x):
        # CIFAR-10 images are (B, 3, 32, 32); flatten to (B, 3072)
        # so they can be processed by linear layers.
        x = torch.flatten(x, start_dim=1)

        # Prunable hidden layer 1 + non-linearity
        x = F.relu(self.fc1(x))

        # Prunable hidden layer 2 + non-linearity
        x = F.relu(self.fc2(x))

        # Final prunable classification layer.
        # No softmax here: CrossEntropyLoss expects raw logits.
        x = self.fc3(x)
        return x
    
    def get_prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3]


# -----------------------
# Small test snippet
# -----------------------
if __name__ == "__main__":
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32)

    model = PrunableNet()
    out = model(x)

    print("Output shape:", out.shape)  # Expected: (batch_size, 10)