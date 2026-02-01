import torch


def evaluate(model, loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for U, Theta, Phi in loader:
            pred = model(U, Theta)
            loss = criterion(pred, Phi)
            total += loss.item() * U.size(0)
    return total / len(loader.dataset)


def train_model(
    model,
    train_loader,
    val_loader=None,
    *,
    epochs=100,
    lr=1e-3,
    device=None,
    verbose=True,
):
    if device is None:
        device = next(model.parameters()).device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0

        for U, Theta, Phi in train_loader:
            optimizer.zero_grad()
            pred = model(U, Theta)
            loss = criterion(pred, Phi)
            loss.backward()
            optimizer.step()
            total += loss.item() * U.size(0)

        train_loss = total / len(train_loader.dataset)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion)
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
