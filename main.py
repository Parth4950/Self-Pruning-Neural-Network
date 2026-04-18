import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import PrunableNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Sparsity Loss
def compute_sparsity_loss(model):
    total_sparsity_loss = 0.0
    for layer in model.get_prunable_layers():
        gates = layer.get_gates()
        total_sparsity_loss += torch.sum(gates) / gates.numel()
    return total_sparsity_loss


# Accuracy Evaluation
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100.0 * correct / total


# Sparsity Percentage
def compute_sparsity_percentage(model, threshold=1e-2):
    total_gates = 0
    pruned_gates = 0

    with torch.no_grad():
        for layer in model.get_prunable_layers():
            gates = layer.get_gates()
            total_gates += gates.numel()
            pruned_gates += (gates < threshold).sum().item()

    return 100.0 * pruned_gates / total_gates


def layerwise_sparsity(model, threshold=1e-2):
    with torch.no_grad():
        for idx, layer in enumerate(model.get_prunable_layers(), start=1):
            gates = layer.get_gates()
            layer_total = gates.numel()
            layer_pruned = (gates < threshold).sum().item()
            layer_percent = 100.0 * layer_pruned / layer_total
            print(f"Layer {idx} sparsity: {layer_percent:.2f}%")


# Main Training Pipeline
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    transform = transforms.ToTensor()
    batch_size = 64

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Lambda experiments
    lambda_values = [1e-3, 1e-2, 1e-1]
    results = []

    for lambda_val in lambda_values:
        print("\n==============================")
        print(f"Training with lambda = {lambda_val}")
        print("==============================")

        model = PrunableNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        num_epochs = 20

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(images)
                classification_loss = criterion(logits, labels)
                sparsity_loss = compute_sparsity_loss(model)

                total_loss = classification_loss + lambda_val * sparsity_loss
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        # Evaluation after training
        test_acc = evaluate_accuracy(model, test_loader, device)
        sparsity = compute_sparsity_percentage(model)

        print(
            f"Lambda: {lambda_val} | Accuracy: {test_acc:.2f}% | Sparsity: {sparsity:.2f}%"
        )
        print("[Insight] Higher lambda should increase sparsity and may reduce accuracy.")
        layerwise_sparsity(model, threshold=1e-2)

        for layer in model.get_prunable_layers():
            gates = layer.get_gates().detach().cpu().numpy()
            print("Min gate:", gates.min(), "Max gate:", gates.max())

        results.append((float(lambda_val), float(test_acc), float(sparsity)))

    # Print Final Results Table
    print("\nFinal Results:")
    print("Lambda\t\tAccuracy\tSparsity")
    for lam, acc, sp in results:
        print(f"{lam}\t{acc:.2f}%\t\t{sp:.2f}%")

    # Save results to CSV
    with open("results.csv", mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["lambda", "accuracy", "sparsity"])
        for lam, acc, sp in results:
            writer.writerow([f"{lam:.8g}", f"{acc:.6f}", f"{sp:.6f}"])

    # Plot 1: Lambda vs Accuracy
    lambdas = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    sparsities = [r[2] for r in results]

    plt.figure()
    plt.plot(lambdas, accuracies, marker="o")
    plt.xscale("log")
    plt.title("Lambda vs Accuracy")
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Plot 2: Lambda vs Sparsity
    plt.figure()
    plt.plot(lambdas, sparsities, marker="o", color="orange")
    plt.xscale("log")
    plt.title("Lambda vs Sparsity")
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Sparsity (%)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


if __name__ == "__main__":
    main()