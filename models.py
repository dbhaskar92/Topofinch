
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_topological.nn import CubicalComplex, PersistenceEntropy
    TOPO_OK = True
except Exception as e:
    TOPO_OK = False
    _ERR = e

class ChannelAttention(nn.Module):
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        z = x.mean(dim=(2,3))       # (B, C)
        z = F.relu(self.fc1(z))     # (B, hidden)
        z = self.fc2(z)             # (B, C)
        w = F.softmax(z, dim=1).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return x * w, w.squeeze(-1).squeeze(-1)  # reweighted x, (B,C)

class TopoHead(nn.Module):
    def __init__(self, maxdim=1):
        super().__init__()
        if not TOPO_OK:
            raise ImportError(f"torch-topological not available: {_ERR}")
        self.cubical = CubicalComplex(maxdim=maxdim)
        self.entropy = PersistenceEntropy()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        feats = []
        for c in range(C):
            dgm = self.cubical(x[:, c, :, :])  # expected (B,H,W)
            f = self.entropy(dgm)              # (B, F) vectorization
            feats.append(f)
        return torch.stack(feats, dim=1)       # (B, C, F)

class TopoClassifier(nn.Module):
    def __init__(self, channels=3, num_classes=5, topo_maxdim=1):
        super().__init__()
        self.attn = ChannelAttention(channels, hidden=32)
        self.topo = TopoHead(maxdim=topo_maxdim)
        self.classifier = None  # lazy init after seeing feature dim
        self.num_classes = num_classes

    def forward(self, x):
        if self.classifier is None:
            with torch.no_grad():
                y, w = self.attn(x)
                z = self.topo(y)  # (B,C,F)
                Fdim = z.shape[-1]
            self.classifier = nn.Sequential(
                nn.Linear(Fdim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, self.num_classes)
            ).to(x.device)
        y, w = self.attn(x)
        z = self.topo(y)                                # (B,C,F)
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        f = (z * w_norm.unsqueeze(-1)).sum(dim=1)       # (B,F)
        logits = self.classifier(f)                     # (B,num_classes)
        return logits, w, f
