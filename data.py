
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from hvc_dataset import HVCDataset

STAGE_TO_IDX = {
    "subsong": 0,
    "protosyll": 1,
    "multi syll": 2,
    "motif": 3,
    "young adult": 4,
}
IDX_TO_STAGE = {v: k for k, v in STAGE_TO_IDX.items()}

def compute_spectrogram_torch(wave, fs, n_fft=1024, hop_length=256, win_length=1024):
    """
    Compute magnitude spectrogram using torch.stft.
    wave: 1D torch.Tensor of shape (samples,)
    returns: torch.Tensor of shape (F, T)
    """
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)  # (1, samples)
    window = torch.hann_window(win_length, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(
        wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True
    )  # (1, F, T)
    mag = spec.abs().squeeze(0)  # (F, T)
    return mag

def freq_band_channels(mag, fs, bands=((0,2000),(2000,4000),(4000,8000))):
    """
    Split a spectrogram into band-limited channels by masking frequencies.
    mag: (F, T); returns (C, F, T) with values zeroed out-of-band for each channel.
    """
    FreqBins = mag.shape[0]
    freqs = torch.linspace(0, fs/2, steps=FreqBins, device=mag.device, dtype=mag.dtype)
    chans = []
    for (lo, hi) in bands:
        mask = (freqs >= lo) & (freqs < hi)
        m = torch.zeros_like(mag)
        m[mask] = mag[mask]
        chans.append(m)
    return torch.stack(chans, dim=0)  # (C, F, T)

class HVCSyllableSpectrogramDataset(Dataset):
    """
    Create (image, label) pairs per selected syllable from HVCDataset.
    Image has shape (C, H, W) where C is the number of frequency bands.
    Label is the vocalization stage index.
    """
    def __init__(self, root, split="train", split_ratio=0.8,
                 bands=((0,2000),(2000,4000),(4000,8000)),
                 n_fft=1024, hop=256, win=1024, target_size=(224,224),
                 stages_filter=None):
        super().__init__()
        self.root = root
        self.bands = bands
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.target_size = target_size

        base = HVCDataset(root=root)

        # Build list of (file_idx, syll_idx) with stage
        items = []
        stages = []
        for i in range(len(base)):
            it = base[i]
            stage = it["meta"]["stage"]
            if stages_filter is not None and stage not in stages_filter:
                continue
            stimes = it["syllable_times"]
            if stimes is None or len(stimes) == 0:
                continue
            for s_idx in range(stimes.shape[0]):
                items.append((i, s_idx))
                stages.append(stage)

        rng = np.random.default_rng(2025)
        items = np.array(items, dtype=object)
        stages = np.array(stages, dtype=object)
        self.ids = []
        if len(items) == 0:
            self.base = base
            return

        # Stratified split by stage
        unique = np.unique(stages)
        train_mask = np.zeros(len(items), dtype=bool)
        for u in unique:
            idxs = np.where(stages == u)[0]
            rng.shuffle(idxs)
            ntr = int(math.floor(split_ratio * len(idxs)))
            train_mask[idxs[:ntr]] = True

        if split == "train":
            self.ids = items[train_mask].tolist()
        else:
            self.ids = items[~train_mask].tolist()

        self.base = base

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_idx, syll_idx = self.ids[idx]
        it = self.base[int(file_idx)]
        fs = float(it["fs"])
        wave = torch.tensor(it["song"], dtype=torch.float32)
        stage = it["meta"]["stage"]
        label = STAGE_TO_IDX.get(stage, -1)

        st, et = it["syllable_times"][int(syll_idx)]
        s = max(0, int(st * fs))
        e = min(len(wave) - 1, int(et * fs))
        if e <= s:
            e = min(len(wave), s + int(0.2 * fs))
        seg = wave[s:e]

        mag = compute_spectrogram_torch(seg, fs, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win)
        mag = torch.log1p(mag)
        chans = freq_band_channels(mag, fs, self.bands)  # (C, F, T)
        img = torch.nn.functional.interpolate(chans.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        img = (img - img.mean(dim=(1,2), keepdim=True)) / (img.std(dim=(1,2), keepdim=True) + 1e-6)

        meta = dict(it["meta"])
        meta["stage_idx"] = label
        meta["syll_idx"] = int(syll_idx)
        return img, label, meta
