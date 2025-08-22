import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import filtfilt, firwin, windows
from torch.utils.data import Dataset

from .utils import load_excel_metadata, parse_filelist

class HVCDataset(Dataset):
    def __init__(self, root, bird=None, stage=None, brain_area=None,
                 neuron_name=None, min_isolation=None):
        self.root = root
        self.excel_path = os.path.join(root, "List_of_HVC_neurons.xlsx")
        self.filelist_path = os.path.join(root, "filelist.txt")

        self.meta = load_excel_metadata(self.excel_path)
        if os.path.exists(self.filelist_path):
            self.filemap = parse_filelist(self.filelist_path)
        else:
            self.filemap = {}

        # filtering metadata if requested
        if bird:
            self.meta = self.meta[self.meta['Bird'] == bird]
        if stage:
            self.meta = self.meta[self.meta['Stage'] == stage]
        if brain_area:
            self.meta = self.meta[self.meta['BrainArea'] == brain_area]
        if neuron_name:
            self.meta = self.meta[self.meta['Neuron'] == neuron_name]
        if min_isolation:
            self.meta = self.meta[self.meta['Isolation'] >= min_isolation]

        self.meta = self.meta.reset_index(drop=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        neuron_name = row["Neuron"]
        brain_area = row["BrainArea"]
        mat_path = os.path.join(self.root, "Analysis_files", brain_area, f"{neuron_name}.mat")
        dbase = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["dbase"].item()

        k = int(row["FileNumber"]) - 1  # MATLAB is 1-based

        # --- song
        fn_song = dbase.SoundFiles[k].name
        mat_song = sio.loadmat(os.path.join(dbase.PathName, fn_song.replace(".dat", ".mat")))
        sound = mat_song["sound"].squeeze()

        # --- neural
        fn_neural = dbase.ChannelFiles[k].name
        mat_neural = sio.loadmat(os.path.join(dbase.PathName, fn_neural.replace(".dat", ".mat")))
        neural = mat_neural["neural"].squeeze()

        fs = float(dbase.Fs)
        neural_plot = neural
        if any("HighPass" in str(f) for f in np.atleast_1d(dbase.EventFunctions)):
            b = firwin(81, 0.034, pass_zero=False, window="hann")
            neural_plot = filtfilt(b, [1.0], neural)

        # --- syllables
        syll_idx = np.array(dbase.SegmentIsSelected[k], dtype=bool)
        seg_times = np.array(dbase.SegmentTimes[k])
        onsets = seg_times[syll_idx, 0] / fs
        offsets = seg_times[syll_idx, 1] / fs
        labels = np.array(dbase.SegmentTitles[k])[syll_idx]

        # --- spikes
        spike_times = np.array([])
        if len(dbase.EventTimes) > 0:
            ev1 = np.array(dbase.EventTimes[0][0, k]).squeeze()
            ev2 = np.array(dbase.EventTimes[0][1, k]).squeeze()
            TempSpikes = ev1 if ev1[0] < ev2[0] else ev2
            mask = np.array(dbase.EventIsSelected[0][1, k], dtype=bool)
            spike_times = TempSpikes[mask] / fs

        return {
            "song": sound.astype(np.float32),
            "neural": neural_plot.astype(np.float32),
            "fs": fs,
            "syllable_times": np.column_stack([onsets, offsets]),
            "syllable_labels": labels,
            "spike_times": spike_times,
            "meta": {
                "bird": row["Bird"],
                "date": row["Date"],
                "age_dph": row.get("Age_dph", None),
                "stage": row["Stage"],
                "brain_area": brain_area,
                "neuron": neuron_name,
                "file_index": k,
            }
        }
