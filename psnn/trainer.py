from __future__ import annotations

from typing import Tuple

import torch


def eval_phi_model(model, loader, criterion) -> float:
    """Evaluate a Phi regressor (PSNN) on a dataloader."""
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    with torch.no_grad():
        for U, Theta, Phi in loader:
            U = U.to(device, non_blocking=True)
            Theta = Theta.to(device, non_blocking=True)
            Phi = Phi.to(device, non_blocking=True)
            pred = model(U, Theta)
            loss = criterion(pred, Phi)
            total += loss.item() * U.size(0)
    return total / len(loader.dataset)


def train_phi_model(
    model,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    device=None,
    verbose: bool = True,
):
    """Train a Phi regressor (PSNN) with MSE loss."""
    if device is None:
        device = next(model.parameters()).device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0

        for U, Theta, Phi in train_loader:
            U = U.to(device, non_blocking=True)
            Theta = Theta.to(device, non_blocking=True)
            Phi = Phi.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(U, Theta)
            loss = criterion(pred, Phi)
            loss.backward()
            optimizer.step()
            total += loss.item() * U.size(0)

        train_loss = total / len(train_loader.dataset)

        if val_loader is not None:
            val_loss = eval_phi_model(model, val_loader, criterion)
            if verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train MSE {train_loss:.6f} | "
                    f"Val MSE {val_loss:.6f}"
                )
        else:
            if verbose:
                print(f"Epoch {epoch:03d} | Train MSE {train_loss:.6f}")

    return model


# Back-compat aliases (prefer the explicit phi names above).
evaluate = eval_phi_model
train_model = train_phi_model


def _make_class_weights(
    labels: torch.Tensor, num_classes: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(labels.detach().cpu(), minlength=int(num_classes)).float()
    weights = counts.sum() / (float(num_classes) * counts.clamp_min(1.0))
    weights[counts == 0] = 0.0
    return weights.to(device), counts


def eval_count_classifier(model, loader, criterion) -> Tuple[float, float]:
    """Evaluate a Theta->count classifier."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for Theta, y in loader:
            Theta = Theta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(Theta)
            loss = criterion(logits, y)
            total_loss += loss.item() * Theta.size(0)
            total_acc += (logits.argmax(dim=1) == y).float().sum().item()
    n = max(1, len(loader.dataset))
    return total_loss / n, total_acc / n


def train_count_classifier(
    model,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    device=None,
    verbose: bool = True,
):
    """Train a Theta->count classifier with class-balanced CrossEntropy."""
    if device is None:
        device = next(model.parameters()).device

    labels = getattr(train_loader.dataset, "Labels", None)
    if labels is None:
        raise AttributeError("train_loader.dataset is missing Labels")

    if hasattr(train_loader.dataset, "num_classes"):
        num_classes = int(getattr(train_loader.dataset, "num_classes"))
    else:
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 0

    class_weights, counts = _make_class_weights(labels, num_classes, device)
    if verbose:
        print(f"Count label counts: {counts.tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        for Theta, y in train_loader:
            Theta = Theta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(Theta)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Theta.size(0)
            total_acc += (logits.argmax(dim=1) == y).float().sum().item()

        n = max(1, len(train_loader.dataset))
        train_loss = total_loss / n
        train_acc = total_acc / n

        if val_loader is not None:
            val_loss, val_acc = eval_count_classifier(model, val_loader, criterion)
            if verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train CE {train_loss:.6f} Acc {train_acc:.4f} | "
                    f"Val CE {val_loss:.6f} Acc {val_acc:.4f}"
                )
        else:
            if verbose:
                print(f"Epoch {epoch:03d} | Train CE {train_loss:.6f} Acc {train_acc:.4f}")

    return model, num_classes


def eval_stability_classifier(model, loader, criterion) -> Tuple[float, float]:
    """Evaluate a (Theta,U)->stable classifier."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for U, Theta, y in loader:
            U = U.to(device, non_blocking=True)
            Theta = Theta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            probs = model(U, Theta).view(-1)
            y_float = y.float()
            loss = criterion(probs, y_float).mean()
            total_loss += loss.item() * U.size(0)
            total_acc += ((probs >= 0.5) == (y_float >= 0.5)).float().sum().item()
    n = max(1, len(loader.dataset))
    return total_loss / n, total_acc / n


def train_stability_classifier(
    model,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    device=None,
    verbose: bool = True,
):
    """Train a stability classifier with class-balanced weighted BCE."""
    if device is None:
        device = next(model.parameters()).device

    labels = getattr(train_loader.dataset, "Labels", None)
    if labels is None:
        raise AttributeError("train_loader.dataset is missing Labels")

    counts = torch.bincount(labels.detach().cpu(), minlength=2).float()
    total = counts.sum().clamp_min(1.0)
    w_pos = (total / (2.0 * counts[1].clamp_min(1.0))).to(device)
    w_neg = (total / (2.0 * counts[0].clamp_min(1.0))).to(device)
    if verbose:
        print(f"Stability label counts: {counts.tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss(reduction="none")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

        for U, Theta, y in train_loader:
            U = U.to(device, non_blocking=True)
            Theta = Theta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            probs = model(U, Theta).view(-1)
            y_float = y.float()
            weights = torch.where(y_float > 0.5, w_pos, w_neg)
            loss = (criterion(probs, y_float) * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * U.size(0)
            total_acc += ((probs >= 0.5) == (y_float >= 0.5)).float().sum().item()

        n = max(1, len(train_loader.dataset))
        train_loss = total_loss / n
        train_acc = total_acc / n

        if val_loader is not None:
            val_loss, val_acc = eval_stability_classifier(model, val_loader, criterion)
            if verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train CE {train_loss:.6f} Acc {train_acc:.4f} | "
                    f"Val CE {val_loss:.6f} Acc {val_acc:.4f}"
                )
        else:
            if verbose:
                print(f"Epoch {epoch:03d} | Train CE {train_loss:.6f} Acc {train_acc:.4f}")

    return model
