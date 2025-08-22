from __future__ import annotations
import os, re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.io import loadmat

def _squeeze(obj):
    if isinstance(obj, np.ndarray) and obj.size == 1:
        return obj.reshape(())
    return obj

def _mat_to_dict(mat_struct) -> Dict[str, Any]:
    out = {}
    for name in mat_struct.dtype.names:
        val = mat_struct[name]
        if isinstance(val, np.ndarray) and val.dtype.names:
            if val.size == 1:
                out[name] = _mat_to_dict(val.reshape(()))
            else:
                out[name] = [ _mat_to_dict(val.reshape((-1,))[i]) for i in range(val.size) ]
        else:
            out[name] = val
    return out

def load_analysis_dbase(path: str) -> Dict[str, Any]:
    m = loadmat(path, squeeze_me=False, struct_as_record=False)
    if 'dbase' not in m:
        raise KeyError(f"'dbase' not found in {path}. Keys: {list(m.keys())}")
    d = m['dbase']
    if hasattr(d, "_fieldnames"):
        d = {fn: getattr(d, fn) for fn in d._fieldnames}
    else:
        d = _mat_to_dict(d)
    Fs = float(np.array(d['Fs']).squeeze())
    def names_from(field):
        arr = d[field]
        names = []
        for i in range(arr.size if isinstance(arr, np.ndarray) else len(arr)):
            x = arr[i] if isinstance(arr, np.ndarray) else arr[i]
            if hasattr(x, 'name'):
                names.append(str(np.array(x.name).squeeze()))
            elif isinstance(x, dict) and 'name' in x:
                names.append(str(np.array(x['name']).squeeze()))
            else:
                names.append(str(np.array(x).squeeze()))
        return names
    sound_files = names_from('SoundFiles')
    neural_files = names_from('ChannelFiles')
    seg_times = [np.array(x).astype(float) for x in d['SegmentTimes'].squeeze()]
    seg_sel   = [np.array(x).astype(bool).squeeze() for x in d['SegmentIsSelected'].squeeze()]
    seg_titles = [np.array(x).astype(object).squeeze() for x in d['SegmentTitles'].squeeze()]
    event_functions = str(np.array(d['EventFunctions']).squeeze()[0])
    event_times = d['EventTimes'].squeeze()
    event_is_selected = d['EventIsSelected'].squeeze()
    path_name = str(np.array(d['PathName']).squeeze())
    times = [np.array(x).squeeze() for x in d['Times'].squeeze()]
    return {
        "Fs": Fs,
        "SoundFiles": sound_files,
        "ChannelFiles": neural_files,
        "SegmentTimes": seg_times,
        "SegmentIsSelected": seg_sel,
        "SegmentTitles": seg_titles,
        "EventFunctions": event_functions,
        "EventTimes": event_times,
        "EventIsSelected": event_is_selected,
        "PathName": path_name,
        "Times": times,
        "Raw": m,
    }

def _load_mat_signal(path: str, key: str) -> np.ndarray:
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key not in m:
        if key == "sound" and "sound" in m:
            return np.asarray(m["sound"]).astype(float)
        if key == "neural" and "neural" in m:
            return np.asarray(m["neural"]).astype(float)
        raise KeyError(f"{key} not found in {path}. Keys: {list(m.keys())}")
    return np.asarray(m[key]).astype(float)

def load_mat_song(path: str) -> np.ndarray:
    return _load_mat_signal(path, "sound")

def load_mat_neural(path: str) -> np.ndarray:
    return _load_mat_signal(path, "neural")

def read_neuron_index(xlsx_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    dfs = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df = df.drop(0).reset_index(drop=True)
        df["BrainArea"] = sheet
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out = out.rename(columns={"Name": "NeuronName", "Ch": "Channel", "Iso": "Isolation", "File": "FileIdx"})
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Bird"] = out["Bird"].astype(str)
    out["Stage"] = out["Stage"].astype(str)
    out["NeuronName"] = out["NeuronName"].astype(str)
    return out

def read_filelist(path_txt: str) -> pd.DataFrame:
    rows = []
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rel = parts[0]
            if not rel.startswith("data/to"):
                continue
            tar = Path(rel).name
            bird = tar.split(".")[0]
            size = None
            if len(parts) >= 2:
                try:
                    size = int(parts[1])
                except Exception:
                    pass
            rows.append({"bird": bird, "tar_path": rel, "size_bytes": size})
    return pd.DataFrame(rows)
