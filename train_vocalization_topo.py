
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import HVCSyllableSpectrogramDataset, STAGE_TO_IDX
from models import TopoClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True, help='Dataset root (contains Analysis_files/, Data/, List_of_HVC_neurons.xlsx, filelist.txt)')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bands', type=str, default='0-2000,2000-4000,4000-8000')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    bands = []
    for bp in args.bands.split(','):
        lo, hi = bp.split('-')
        bands.append((float(lo), float(hi)))

    train_ds = HVCSyllableSpectrogramDataset(root=args.root, split='train', bands=bands)
    test_ds  = HVCSyllableSpectrogramDataset(root=args.root, split='test',  bands=bands)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TopoClassifier(channels=len(bands), num_classes=len(STAGE_TO_IDX)).to(args.device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0.0, 0
        for x, y, _ in train_ld:
            x = x.to(args.device)
            y = torch.tensor(y, dtype=torch.long, device=args.device)
            logits, attn, feats = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0)
            tot_correct += (logits.argmax(dim=1) == y).float().sum().item()
            tot_n += x.size(0)
        tr_loss = tot_loss / max(1, tot_n)
        tr_acc = tot_correct / max(1, tot_n)

        model.eval()
        tot_loss, tot_correct, tot_n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y, _ in test_ld:
                x = x.to(args.device)
                y = torch.tensor(y, dtype=torch.long, device=args.device)
                logits, attn, feats = model(x)
                loss = crit(logits, y)
                tot_loss += loss.item() * x.size(0)
                tot_correct += (logits.argmax(dim=1) == y).float().sum().item()
                tot_n += x.size(0)
        te_loss = tot_loss / max(1, tot_n)
        te_acc = tot_correct / max(1, tot_n)

        print(f'[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | test loss {te_loss:.4f} acc {te_acc:.3f}')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/topo_classifier.pt')
    print('Saved to checkpoints/topo_classifier.pt')

if __name__ == '__main__':
    main()
