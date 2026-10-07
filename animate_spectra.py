import glob, re, sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

folder = sys.argv[1] if len(sys.argv) > 1 else "ts_big/live/epochs"
out    = sys.argv[2] if len(sys.argv) > 2 else "spectra_training.gif"
files  = sorted(glob.glob(f"{folder}/spectra_epoch_*.csv"))
frames = [pd.read_csv(f) for f in files]
epochs = [int(re.search(r"epoch_(\d+)", f).group(1)) for f in files]
ids = [c[5:] for c in frames[0].columns if c.startswith("true_")]
wl = frames[0]["wavelength_nm"]

nc = min(3, len(ids)); nr = -(-len(ids) // nc)
fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 3.2 * nr), squeeze=False)
fits = []
for ax, i in zip(axs.ravel(), ids):
    ax.plot(wl, frames[0][f"data_{i}"], color="0.75", lw=.7, label="noisy input")
    ax.plot(wl, frames[0][f"true_{i}"], "k", lw=1, label="true")
    fits.append(ax.plot(wl, frames[0][f"fit_{i}"], "r", lw=1, label="CNN fit")[0])
    ax.set_title(f"#{i}", fontsize=9); ax.set_xlabel("nm")
for ax in axs.ravel()[len(ids):]: ax.axis("off")
axs.ravel()[0].legend(fontsize=7)
title = fig.suptitle(""); plt.tight_layout()

def update(k):
    for ln, i in zip(fits, ids): ln.set_ydata(frames[k][f"fit_{i}"])
    title.set_text(f"epoch {epochs[k]}")
    return fits

FuncAnimation(fig, update, frames=len(frames)).save(out, writer=PillowWriter(fps=3))
print(f"{len(frames)} epochs -> {out}")
