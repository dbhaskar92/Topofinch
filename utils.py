import pandas as pd

def load_excel_metadata(excel_path):
    # Combine all three sheets into one dataframe
    xl = pd.ExcelFile(excel_path)
    dfs = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df["BrainArea"] = sheet
        dfs.append(df)
    meta = pd.concat(dfs, ignore_index=True)
    # Standardize column names if necessary
    if "Neuron name" in meta.columns:
        meta = meta.rename(columns={"Neuron name": "Neuron"})
    if "Bird name" in meta.columns:
        meta = meta.rename(columns={"Bird name": "Bird"})
    if "File number" in meta.columns:
        meta = meta.rename(columns={"File number": "FileNumber"})
    return meta

def parse_filelist(filelist_path):
    filemap = {}
    with open(filelist_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                key, val = parts
                filemap[key] = val
    return filemap
