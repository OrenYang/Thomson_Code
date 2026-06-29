"""
Batch-export every fiber from every Thomson .pkl shot in a folder into
individual CSVs (wavelength_nm, intensity, sigma, gamma, r_um).

Each .pkl gets its own subfolder (named after the shot) inside the output
folder, containing one CSV per fiber.

You MUST edit the import line below to point at the actual module where
your Thomson/Bin/Calfile classes are defined (the same file that has the
`sys.modules['thomson_analysis_preprocess'] = sys.modules[__name__]` alias
trick at the top), or pickle.load will fail to find the classes.

Usage:
    python pkl_to_csv_batch.py /path/to/pkl_folder /path/to/output_folder

If no arguments are given, folder dialogs will open instead.
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
from tkinter import filedialog, Tk

# EDIT THIS to match your actual analysis module's filename (without .py)
from thomson_analysis import Thomson  # noqa: F401


def export_pkl(pkl_path, out_folder):
    ts = Thomson(pkl_path)
    has_cal = hasattr(ts, 'cal')
    if not has_cal:
        print(f"  Warning: {ts.name} has no calibration (.cal) attached; sigma/gamma will be NaN.")

    # each shot gets its own subfolder
    shot_folder = os.path.join(out_folder, ts.name)
    os.makedirs(shot_folder, exist_ok=True)

    for b in ts.bins:
        if not hasattr(b, 'wavelength'):
            print(f"  Fiber {b.fiber}: not calibrated (no wavelength array), skipping.")
            continue
        sigma = np.nan
        gamma = np.nan
        if has_cal:
            cal_bin = next((c for c in ts.cal.bins if c.fiber == b.fiber), None)
            if cal_bin is not None and hasattr(cal_bin, 's'):
                sigma = cal_bin.s
                gamma = cal_bin.g
        r_um = getattr(b, 'r', '')
        df = pd.DataFrame({
            'wavelength_nm': b.wavelength,
            'intensity':     b.spectrum,
        })
        df['sigma'] = sigma
        df['gamma'] = gamma
        df['r_um'] = r_um
        out_path = os.path.join(shot_folder, f'{ts.name}_fiber{b.fiber}.csv')
        df.to_csv(out_path, index=False)
        print(f"  Saved fiber {b.fiber} -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pkl_folder = sys.argv[1]
        out_folder = sys.argv[2]
    else:
        Tk().withdraw()
        pkl_folder = filedialog.askdirectory(title="Select folder containing .pkl files")
        out_folder = filedialog.askdirectory(title="Select output folder for CSVs")

    os.makedirs(out_folder, exist_ok=True)
    pkl_files = sorted(glob.glob(os.path.join(pkl_folder, '*.pkl')))
    print(f"Found {len(pkl_files)} pkl file(s) in {pkl_folder}")

    for pkl_path in pkl_files:
        print(f"Processing {pkl_path}")
        export_pkl(pkl_path, out_folder)
