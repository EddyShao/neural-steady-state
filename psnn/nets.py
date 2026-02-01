import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, width, depth, out_dim):
        super().__init__()
        layers = []
        dims = [in_dim] + [width]*depth + [out_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1])]
            layers += [nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class PSNN(nn.Module):
    def __init__(self, dim_theta, dim_u, embed_dim=8, width=[30, 20], depth=[4, 3], eta=0.01):
        super().__init__()
        if isinstance(width, int):
            width = [width] * 2
        if isinstance(depth, int):
            depth = [depth] * 2
        self.pnn = MLP(dim_theta, width[0], depth[0], embed_dim)
        self.snn = MLP(dim_u,     width[1], depth[1], embed_dim) 
        self.eta = eta

    def forward(self, U, Theta):
        # U: (B, n), Theta: (B, m) or (1, m) broadcastable
        p = self.pnn(Theta)                 # (B, N)
        s = self.snn(U)                     # (B, N)
        y = torch.sum(p*s, dim=-1, keepdim=True)   # (B,1)
        # return -self.eta + (1 + 2*self.eta) * torch.sigmoid(y)  # in (eta, 1+eta)
        return torch.sigmoid(y) + self.eta * torch.tanh(y/2)  # in (-eta, 1+eta) 


class PSNN_stb(nn.Module):
    def __init__(self, dim_theta, dim_u, embed_dim=8, width=[30, 20], depth=[4, 3], eta=0.01):
        super().__init__()
        if isinstance(width, int):
            width = [width] * 2
        if isinstance(depth, int):
            depth = [depth] * 2
        self.pnn = MLP(dim_theta, width[0], depth[0], embed_dim)
        self.snn = MLP(dim_u,     width[1], depth[1], embed_dim)
        self.eta = eta

    def forward(self, U, Theta):
        # U: (B, n), Theta: (B, m) or (1, m) broadcastable
        p = self.pnn(Theta)                 # (B, N)
        s = self.snn(U)                     # (B, N)
        y = torch.sum(p*s, dim=-1, keepdim=True)   # (B,1)
        return (self.eta + 1.0) * torch.tanh(y)


class ThetaCountClassifier(nn.Module):
    """Classify number of solutions from parameters only."""
    def __init__(self, dim_theta, num_classes, width=32, depth=3):
        super().__init__()
        self.net = MLP(dim_theta, width, depth, num_classes)

    def forward(self, Theta):
        return self.net(Theta)


class StabilityClassifier(nn.Module):
    """Classify stability from (theta, u)."""
    def __init__(self, dim_theta, dim_u, embed_dim=16, width=[32, 32], depth=[2, 2]):
        super().__init__()
        if isinstance(width, int):
            width = [width] * 2
        if isinstance(depth, int):
            depth = [depth] * 2
        self.theta_net = MLP(dim_theta, width[0], depth[0], embed_dim)
        self.u_net = MLP(dim_u, width[1], depth[1], embed_dim)

    def forward(self, U, Theta):
        t = self.theta_net(Theta)
        u = self.u_net(U)
        y = torch.sum(t * u, dim=-1, keepdim=True)
        return torch.sigmoid(y)
