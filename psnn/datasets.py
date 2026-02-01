# psnn_dataset.py
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader

class PSNNDataset(Dataset):
    """
    Simple PSNN dataset for Gray–Scott data (no normalization).
    Expects .npz with keys:
        - "Theta": (N, m)
        - "U":     (N, n)
        - "Phi":   (N,)
    Converts everything to torch tensors once on load.
    """
    def __init__(self, npz_path: str, device: str = None):
        super().__init__()
        data = np.load(npz_path)
        Theta = data["Theta"].astype(np.float32)
        U     = data["U"].astype(np.float32)
        Phi   = data["Phi"].astype(np.float32)

        assert Theta.shape[0] == U.shape[0] == Phi.shape[0], "Size mismatch among arrays"

        # Convert to tensors immediately
        self.Theta = torch.from_numpy(Theta)
        self.U     = torch.from_numpy(U)
        self.Phi   = torch.from_numpy(Phi)[:, None]   # (N,1) for model output alignment

        # Optional: move to device (CPU or CUDA)
        if device is not None:
            self.Theta = self.Theta.to(device)
            self.U     = self.U.to(device)
            self.Phi   = self.Phi.to(device)

    def __len__(self):
        return self.U.shape[0]

    def __getitem__(self, idx: int):
        # Direct tensor indexing—already on device
        return self.U[idx], self.Theta[idx], self.Phi[idx]


def make_loaders(train_npz, test_npz, batch_size=1024, num_workers=0, device=None):
    train_ds = PSNNDataset(train_npz, device=device)
    test_ds  = PSNNDataset(test_npz, device=device)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, drop_last=False,
                              num_workers=num_workers)
    return train_loader, test_loader


class ThetaCountDataset(Dataset):
    """Dataset of (Theta, count) from observation pickles."""
    def __init__(self, obs_path: str, device: str = None):
        super().__init__()
        obs = joblib.load(obs_path)
        thetas = []
        raw_counts = []
        for entry in obs:
            thetas.append(np.asarray(entry["Theta"], dtype=np.float32))
            raw_counts.append(int(len(entry.get("U", []))))

        raw_counts = np.asarray(raw_counts, dtype=np.int64)
        # Map possibly non-contiguous counts (e.g. {2,4}) to contiguous class IDs {0,1}.
        class_values = np.unique(raw_counts)
        count_to_class = {int(v): int(i) for i, v in enumerate(class_values.tolist())}
        labels = np.asarray([count_to_class[int(v)] for v in raw_counts.tolist()], dtype=np.int64)

        self.Theta = torch.from_numpy(np.asarray(thetas, dtype=np.float32))
        self.Labels = torch.from_numpy(labels)
        # For decoding predicted classes back to the original count.
        self.ClassValues = torch.from_numpy(class_values.astype(np.int64))
        self.num_classes = int(class_values.shape[0])

        if device is not None:
            self.Theta = self.Theta.to(device)
            self.Labels = self.Labels.to(device)
            self.ClassValues = self.ClassValues.to(device)

    def __len__(self):
        return self.Theta.shape[0]

    def __getitem__(self, idx: int):
        return self.Theta[idx], self.Labels[idx]


class ThetaStabilityDataset(Dataset):
    """Dataset of (U, Theta, stable_label) from observation pickles."""
    def __init__(self, obs_path: str, device: str = None):
        super().__init__()
        obs = joblib.load(obs_path)
        thetas = []
        us = []
        labels = []
        for entry in obs:
            theta = np.asarray(entry["Theta"], dtype=np.float32)
            for sol in entry.get("U", []):
                u = np.asarray(sol.get("u"), dtype=np.float32)
                stable = bool(sol.get("stable", False))
                thetas.append(theta)
                us.append(u)
                labels.append(1 if stable else 0)

        self.Theta = torch.from_numpy(np.asarray(thetas, dtype=np.float32))
        self.U = torch.from_numpy(np.asarray(us, dtype=np.float32))
        self.Labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

        if device is not None:
            self.Theta = self.Theta.to(device)
            self.U = self.U.to(device)
            self.Labels = self.Labels.to(device)

    def __len__(self):
        return self.U.shape[0]

    def __getitem__(self, idx: int):
        return self.U[idx], self.Theta[idx], self.Labels[idx]


def make_obs_loaders(train_obs, test_obs, batch_size=1024, num_workers=0, device=None, mode="count"):
    if mode == "count":
        train_ds = ThetaCountDataset(train_obs, device=device)
        test_ds = ThetaCountDataset(test_obs, device=device)
    elif mode == "stability":
        train_ds = ThetaStabilityDataset(train_obs, device=device)
        test_ds = ThetaStabilityDataset(test_obs, device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=False,
                              num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=num_workers)
    return train_loader, test_loader
