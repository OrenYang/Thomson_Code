#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ts_train_big.py  -  train ONE large CNN for Thomson-scattering (ion-acoustic) spectra of Ne / H / Xe plasmas,
then use it to analyse all your spectra.

WHAT IT DOES
  * Simulates a very large synthetic data set (default 1 000 000 spectra, written to disk in shards, resumable).
  * Parameter values are drawn AROUND your fitted radial profile (out5_averaged.csv):
        60 % "profile"  : a random point on the fibre profile (interpolated between fibres) + jitter
        30 % "envelope" : uniform over the min/max of the profile, widened
        10 % "wide"     : uniform over your full fit bounds (robustness)
  * The ionization states are randomised too: Ne-20 Z in NE_Z, Xe-132 Z in XE_Z (H is always H+).
    The CNN predicts Z_Ne and Z_Xe as probabilities, plus every fit parameter with an uncertainty.
  * Instrument function (Voigt sigma, gamma) is randomised and given to the CNN as an extra input, so one
    model works for fibres/setups with slightly different instrument widths.
  * Any spectrum CSV is resampled to the training wavelength grid before it is analysed, so pixel grids may differ.

USAGE
    python ts_train_big.py all                          # generate data + train + evaluate (default)
    python ts_train_big.py generate | train | evaluate  # the individual steps (generate is resumable)
    python ts_train_big.py train --resume               # continue training from the last checkpoint
    python ts_train_big.py analyze "spectra/*.csv"      # CNN analysis of all your spectra (fast)
    python ts_train_big.py analyze "spectra/*.csv" --refine   # + lmfit differential-evolution fit with CNN-narrowed bounds
    python ts_train_big.py selftest                     # check forward model against plasmapy's lmfit model, timing

Spectrum CSV columns: wavelength_nm, intensity  (+ optional sigma, gamma in metres, as in your other notebook).
Needs: pip install plasmapy lmfit tensorflow scipy joblib matplotlib pandas
"""
import os, sys, time, json, glob, copy, hashlib, argparse, shutil, warnings
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from lmfit import Parameters
import astropy.units as u
from plasmapy.diagnostics import thomson
from plasmapy.particles import Particle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
warnings.filterwarnings("ignore", message=".*instrument function.*")

# ============================================================================================================
#                                              USER SETTINGS
# ============================================================================================================
OUT_DIR       = "ts_big"                 # everything is written here (data shards, model, plots, meta.json)
PROFILE_CSV   = "out5_averaged.csv"      # your fitted profile: values to centre the training data on

# ---- wavelength grid / instrument. If you have a real spectrum CSV, put its path in GRID_FROM_CSV: grid, sigma and
#      gamma are then taken from it. Otherwise the numbers below are used (same defaults as your other notebook).
GRID_FROM_CSV = ""                       # e.g. "fiber8_spectrum.csv"
WL_MIN_NM, WL_MAX_NM, N_PIX = 524.0, 529.0, 1024
INSTR_CENTER  = (1.0e-11, 1.0e-11)       # Voigt (sigma, gamma) in METRES
INSTR_SPREAD  = 2.0                      # training draws sigma, gamma log-uniformly in [centre/spread, centre*spread]
                                         # (1.0 = instrument function fixed; the CNN then ignores the instrument input)
PROBE_NM, SCATTER_DEG = 526.5, 90.0

# ---- species and ionization. H is always H+ (Z=1). Ne-20 and Xe-132 charge states are drawn from these lists.
NE_Z = list(range(6, 11))                # Ne 6+ ... 10+ (10+ = bare nucleus)
XE_Z = list(range(10, 27))               # Xe 10+ ... 26+
NE_Z_NOMINAL, XE_Z_NOMINAL = None, None  # None -> read from PROFILE_CSV (ion_z_0 / ion_z_2 = 9 and 17 in your file)
Z_NOMINAL_BOOST = 0.30                   # fraction of samples that use the nominal Z; the rest are uniform over the lists
XE_TO_H = 0.0065                         # ifract_Xe / ifract_H, fixed as in your fits

# ---- fit bounds / starting values: copied from your notebook ('n' in m^-3, T in eV, speeds in m/s)
FIT_BOUNDS = {"n": (1e23, 1e27), "T_e_0": (0.5, 800.), "T_i_0": (10., 5000.), "T_i_1": (1., 500.),
              "ion_speed_0": (-5e5, 5e5), "ion_speed_1": (-5e5, 5e5), "ion_speed_2": (-5e5, 5e5),
              "electron_speed_0": (-5e5, 5e5), "ifract_0": (0., 0.6)}
INIT_VALUES = {"n": 5e23, "T_e_0": 50., "T_i_0": 150., "T_i_1": 150., "ion_speed_0": 0., "ion_speed_1": 0.,
               "ion_speed_2": 0., "electron_speed_0": 0., "ifract_0": 0.4}
LOG_PARAMS  = ["n", "T_e_0", "T_i_0", "T_i_1"]            # CNN output is scaled in log10 for these (linear for the rest)
SAMPLE_LOG  = LOG_PARAMS + ["ifract_0"]                   # these are drawn log-uniformly / log-normally
LOG_FLOOR   = 1e-3                                        # lower limit for log-sampling a parameter whose bound is 0

# ---- how the training parameters are drawn around the profile
MIX          = (0.60, 0.30, 0.10)       # profile-anchored / widened envelope / full fit bounds
JITTER_FLOOR = {"n": 0.35, "T_e_0": 0.20, "T_i_0": 0.20, "T_i_1": 0.30,        # decades for log params,
                "ion_speed_0": 5e4, "ion_speed_1": 6e4, "ion_speed_2": 6e4,    # m/s for speeds
                "electron_speed_0": 0.0, "ifract_0": 0.15}                     # (electron speed is always uniform)
ENV_PAD_DEC  = 0.40                     # envelope padding: decades (log params) ...
ENV_PAD_FRAC = 0.40                     # ... or fraction of the profile range (linear params)

# ---- data set size
N_SPECTRA  = 1_000_000                  # TRAINING spectra (disk: about 4 kB each, i.e. ~4 GB per million)
N_VAL, N_TEST = 20_000, 20_000          # held-out spectra (validation during training / final evaluation)
SHARD_SIZE = 51_200                     # spectra per file on disk (rounded to a multiple of BATCH)
N_JOBS     = -1                         # CPU cores for generation
SEED       = 42

# ---- training
ARCH            = "medium"              # "light" (~0.4 M params, CPU ok) | "medium" (~1 M, GPU advised) | "paper" (1.4 M, slow)
EPOCHS          = 60                    # one epoch = one pass over all training spectra
BATCH, LR       = 512, 5e-4
NOISE_MAX_FRAC  = 0.05                  # training noise ~U(0,x)*spectrum max, redrawn every batch. Use >= your real noise.
Z_LOSS_WEIGHT   = 1.0                   # weight of the charge-state classification loss
PATIENCE        = 8                     # early-stopping patience (epochs)
TIME_BUDGET_MIN = None                  # stop training after this many minutes (None = no limit)
LIVE_PLOTS      = True                  # redraw 4 progress figures in <OUT_DIR>/live/ after every epoch
LIVE_EVERY, LIVE_N = 1, 3000            # ... every LIVE_EVERY epochs, using LIVE_N noisy validation spectra
LIVE_SPECTRA     = 6                    # number of simulated spectra shown with the CNN's fit (05_spectra.png), archived every epoch

# ---- analysis / refinement fit (same logic as your notebook)
R2_TARGET, NARROW_MAX_NFEV, MIN_WIDTH = 0.98, 6000, 0.10
REFINE_TOP_Z, REFINE_TRIES = 3, 2       # charge-state combinations tried by --refine, DE tries per combination
# ============================================================================================================

for _k in ("OUT_DIR", "N_SPECTRA", "N_VAL", "N_TEST", "SHARD_SIZE", "BATCH", "EPOCHS", "N_JOBS", "ARCH", "PROFILE_CSV"):
    if "TS_" + _k in os.environ: globals()[_k] = type(globals()[_k])(os.environ["TS_" + _k])
if os.environ.get("TS_BUDGET"): TIME_BUDGET_MIN = float(os.environ["TS_BUDGET"])

EV_TO_K = 11604.51812155
PARAM_NAMES = ["n", "T_e_0", "T_i_0", "T_i_1", "ion_speed_0", "ion_speed_1", "ion_speed_2", "electron_speed_0", "ifract_0"]
NP, NN_ = len(PARAM_NAMES), len(PARAM_NAMES) - 1       # 9 parameters: 8 normal heads + ifract_0 (beta head, must be last)
UNIT_FMT = {"n": (1e-24, "1e18 cm^-3"), "T_e_0": (1, "eV"), "T_i_0": (1, "eV"), "T_i_1": (1, "eV"),
            "ion_speed_0": (1e-3, "km/s"), "ion_speed_1": (1e-3, "km/s"), "ion_speed_2": (1e-3, "km/s"),
            "electron_speed_0": (1e-3, "km/s"), "ifract_0": (1, "")}
MODE_NAMES = ["profile", "envelope", "wide"]


# ------------------------------------------------------------------------------------------------ I/O helpers
def load_ts_csv(path):
    """Spectrum CSV -> wavelength [nm], intensity (raw), sigma [m] or None, gamma [m] or None."""
    df = pd.read_csv(path)
    wl = df["wavelength_nm"].to_numpy(float); y = df["intensity"].to_numpy(float)
    good = np.isfinite(wl) & np.isfinite(y)
    if not good.all(): print(f"  note: {path}: dropped {(~good).sum()} non-finite rows")
    wl, y = wl[good], y[good]; o = np.argsort(wl)
    s = float(df["sigma"].iloc[0]) if "sigma" in df and np.isfinite(df["sigma"].iloc[0]) else None
    g = float(df["gamma"].iloc[0]) if "gamma" in df and np.isfinite(df["gamma"].iloc[0]) else None
    return wl[o], y[o], s, g


def load_profile(path):
    """Fitted radial profile (your out*_averaged.csv): per-fibre mean and std of every CNN parameter."""
    if not path or not Path(path).exists(): return None
    d = pd.read_csv(path)
    if "radius_um" in d: d = d.sort_values("radius_um")
    M = np.stack([d[f"{k}_mean"].to_numpy(float) for k in PARAM_NAMES], 1)
    S = np.stack([d[f"{k}_std"].to_numpy(float) for k in PARAM_NAMES], 1)
    ok = np.all(np.isfinite(M), 1)
    zz = lambda c: float(d[c].iloc[0]) if c in d else None
    return {"mean": M[ok].tolist(), "std": np.nan_to_num(S[ok]).tolist(), "ne_z": zz("ion_z_0_mean"),
            "xe_z": zz("ion_z_2_mean"), "source": str(path)}


def make_cfg():
    """Collect all settings into one dict. It is saved to meta.json, so `analyze` needs nothing else."""
    prof = load_profile(PROFILE_CSV)
    if prof is None: print(f"WARNING: profile CSV '{PROFILE_CSV}' not found -> training data drawn over the full fit bounds only")
    grid = {"wl_min_nm": float(WL_MIN_NM), "wl_max_nm": float(WL_MAX_NM), "n_pix": int(N_PIX)}
    center = [float(INSTR_CENTER[0]), float(INSTR_CENTER[1])]
    if GRID_FROM_CSV:
        wl, _, s, g = load_ts_csv(GRID_FROM_CSV)
        grid = {"wl_min_nm": float(wl[0]), "wl_max_nm": float(wl[-1]), "n_pix": int(len(wl))}
        if s and g: center = [s, g]
    rnd = lambda z: int(round(z)) if z is not None else None
    nom = [NE_Z_NOMINAL if NE_Z_NOMINAL is not None else rnd(prof and prof["ne_z"]),
           XE_Z_NOMINAL if XE_Z_NOMINAL is not None else rnd(prof and prof["xe_z"])]
    shard = max(BATCH, SHARD_SIZE // BATCH * BATCH)
    data = dict(version=1, grid=grid, probe_nm=PROBE_NM, scatter_deg=SCATTER_DEG, ne_z=list(NE_Z), xe_z=list(XE_Z),
                xe_to_h=XE_TO_H, fit_bounds={k: list(v) for k, v in FIT_BOUNDS.items()}, log_params=list(LOG_PARAMS),
                sample_log=list(SAMPLE_LOG), log_floor=LOG_FLOOR, mix=list(MIX), jitter_floor=JITTER_FLOOR,
                env_pad_dec=ENV_PAD_DEC, env_pad_frac=ENV_PAD_FRAC, instr_center=center, instr_spread=INSTR_SPREAD,
                z_nominal=nom, z_nominal_boost=Z_NOMINAL_BOOST, profile=prof, shard_size=shard)
    tag = hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return dict(data, tag=tag, seed=SEED, noise_max_frac=NOISE_MAX_FRAC, arch=ARCH, init_values=INIT_VALUES,
                r2_target=R2_TARGET, narrow_max_nfev=NARROW_MAX_NFEV, min_width=MIN_WIDTH, refine_top_z=REFINE_TOP_Z,
                refine_tries=REFINE_TRIES, n_spectra=N_SPECTRA, n_val=N_VAL, n_test=N_TEST, batch=BATCH, lr=LR,
                epochs=EPOCHS, z_loss_weight=Z_LOSS_WEIGHT, patience=PATIENCE)


# ------------------------------------------------------------------------------------------------ physics / scaling
def _voigt(w, sigma, gamma):
    # imported locally so the function pickles cleanly when joblib ships it to worker processes
    from scipy.special import voigt_profile
    return voigt_profile(w, sigma, gamma)


class Ctx:
    """Everything that depends on the configuration: wavelength grid, ions, forward model, parameter scaling."""

    def __init__(self, cfg):
        self.cfg = cfg; g = cfg["grid"]
        self.wl_nm = np.linspace(g["wl_min_nm"], g["wl_max_nm"], int(g["n_pix"]))
        self.n_wl, self.wl_m, self.probe_m = len(self.wl_nm), self.wl_nm * 1e-9, cfg["probe_nm"] * 1e-9
        a = np.deg2rad(cfg["scatter_deg"])
        self.pv, self.sv = np.array([1., 0., 0.]), np.array([np.cos(a), np.sin(a), 0.])
        self.khat = (self.sv - self.pv) / np.linalg.norm(self.sv - self.pv)
        self.ne_z, self.xe_z = [int(z) for z in cfg["ne_z"]], [int(z) for z in cfg["xe_z"]]
        mass = lambda s: Particle(s).mass.to(u.kg).value
        self.ne_m = np.array([mass(f"Ne-20 {z}+") for z in self.ne_z])
        self.xe_m = np.array([mass(f"Xe-132 {z}+") for z in self.xe_z])
        self.h_m, self.xe_to_h = mass("H+"), cfg["xe_to_h"]
        span = self.wl_m.max() - self.wl_m.min()
        self.w_grid = np.linspace(-span / 2, span / 2, self.n_wl)      # zero-centred grid, as spectral_density_model builds it
        self.fb = {k: tuple(v) for k, v in cfg["fit_bounds"].items()}
        self.log = set(cfg["log_params"])

    # ---- forward model (max-normalised spectrum, instrument function applied, T_i_2 = T_i_1, Xe tied to H)
    def instr_arr(self, sigma, gamma):
        v = _voigt(self.w_grid, sigma, gamma); v = v / v.max(); return v / v.sum()

    def forward(self, th, zne_i, zxe_i, sigma, gamma):
        n, Te, Ti0, Ti1, v0, v1, v2, ve, f0 = th; r = self.xe_to_h
        ifr = np.array([f0, (1 - f0) / (1 + r), (1 - f0) * r / (1 + r)])
        Ti = np.array([Ti0, Ti1, Ti1]) * EV_TO_K
        z = np.array([self.ne_z[zne_i], 1., self.xe_z[zxe_i]], float)
        m = np.array([self.ne_m[zne_i], self.h_m, self.xe_m[zxe_i]])
        iv = np.array([v0, v1, v2])[:, None] * self.khat[None, :]
        _, S = thomson.spectral_density_lite(self.wl_m, self.probe_m, n, np.array([Te * EV_TO_K]), Ti, np.array([1.0]),
                                             ifr, z, m, (ve * self.khat)[None, :], iv, self.pv, self.sv,
                                             instr_func_arr=self.instr_arr(sigma, gamma))
        return S / S.max()

    def ions(self, zi, zj): return [f"Ne-20 {self.ne_z[zi]}+", "H+", f"Xe-132 {self.xe_z[zj]}+"]

    def settings(self, zi, zj, sigma, gamma):
        def instr_func(w):
            v = _voigt(w.to(u.m).value, sigma, gamma); return v / v.max()
        return {"probe_wavelength": self.probe_m, "probe_vec": self.pv, "scatter_vec": self.sv, "ions": self.ions(zi, zj),
                "ion_vdir": np.tile(self.khat, (3, 1)), "electron_vdir": np.tile(self.khat, (1, 1)), "instr_func": instr_func}

    # ---- scaling of the CNN targets to [0, 1]
    def to_unit(self, TH):
        U = np.empty(TH.shape, float)
        for j, k in enumerate(PARAM_NAMES):
            lo, hi = self.fb[k]; v = TH[:, j]
            U[:, j] = (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if k in self.log else (v - lo) / (hi - lo)
        return U

    def from_unit(self, U):
        TH = np.empty(U.shape, float)
        for j, k in enumerate(PARAM_NAMES):
            lo, hi = self.fb[k]; x = np.clip(U[:, j], 0, 1)
            TH[:, j] = 10 ** (np.log10(lo) + x * (np.log10(hi) - np.log10(lo))) if k in self.log else lo + x * (hi - lo)
        return TH

    def scale_instr(self, sigma, gamma):
        """(sigma, gamma) in metres -> the CNN's 2 side inputs (log scale, 0..1 over the training range)."""
        sigma, gamma = np.atleast_1d(sigma).astype(float), np.atleast_1d(gamma).astype(float)
        s, c = self.cfg["instr_spread"], self.cfg["instr_center"]
        if s <= 1: return np.full((len(sigma), 2), 0.5, np.float32)
        a = 2 * np.log10(s)
        return np.stack([(np.log10(sigma) - np.log10(c[0])) / a + .5, (np.log10(gamma) - np.log10(c[1])) / a + .5], 1).astype(np.float32)


_CTX_CACHE = {}
def get_ctx(cfg):
    if cfg["tag"] not in _CTX_CACHE: _CTX_CACHE.clear(); _CTX_CACHE[cfg["tag"]] = Ctx(cfg)
    return _CTX_CACHE[cfg["tag"]]


# ------------------------------------------------------------------------------------------------ parameter sampler
class Sampler:
    """Draws (parameters, charge states, instrument widths) around the fitted profile; see the module docstring."""

    def __init__(self, cx):
        self.cx, cfg = cx, cx.cfg
        self.islog = np.array([k in set(cfg["sample_log"]) for k in PARAM_NAMES])
        lo = np.array([cx.fb[k][0] for k in PARAM_NAMES], float); hi = np.array([cx.fb[k][1] for k in PARAM_NAMES], float)
        lo = np.where(self.islog & (lo <= 0), cfg["log_floor"], lo)
        self.lo_p, self.hi_p = lo, hi
        self.lo_w, self.hi_w = self._w(lo), self._w(hi)
        self.mix = np.array(cfg["mix"], float)
        prof = cfg.get("profile"); self.PW = None
        if prof:
            M = np.clip(np.array(prof["mean"], float), lo, hi); S = np.array(prof["std"], float)
            self.PW = self._w(M)
            rel = np.where(M > 0, S / np.where(M > 0, M, 1), 0)          # fit-to-fit scatter in the CSV, in work space
            est = np.where(self.islog, np.median(rel, 0) / np.log(10), np.median(S, 0))
            self.jit = np.maximum(np.array([cfg["jitter_floor"][k] for k in PARAM_NAMES], float), est)
            rngw = self.PW.max(0) - self.PW.min(0)
            pad = np.where(self.islog, cfg["env_pad_dec"], cfg["env_pad_frac"] * rngw)
            self.env_lo = np.maximum(self.PW.min(0) - pad, self.lo_w); self.env_hi = np.minimum(self.PW.max(0) + pad, self.hi_w)
        else:
            self.mix = np.array([0., 0., 1.])
        self.j_ve = PARAM_NAMES.index("electron_speed_0")

    def _w(self, P):                                   # physical -> work space (log10 for SAMPLE_LOG params)
        W = np.array(P, float, copy=True); W[..., self.islog] = np.log10(W[..., self.islog]); return W

    def _reflect(self, W):
        # fold samples that leave the fit bounds back inside (clipping would pile probability on the bounds)
        W = np.where(W > self.hi_w, 2 * self.hi_w - W, W); W = np.where(W < self.lo_w, 2 * self.lo_w - W, W)
        return np.clip(W, self.lo_w, self.hi_w)

    def sample(self, n, rg):
        mode = rg.choice(3, size=n, p=self.mix / self.mix.sum()).astype(np.int8)
        W = rg.uniform(self.lo_w, self.hi_w, (n, NP))                              # mode 2: full fit bounds
        iA, iB = np.where(mode == 0)[0], np.where(mode == 1)[0]
        if len(iA):                                                                # mode 0: profile + jitter
            nf = len(self.PW)
            if nf == 1: C = np.repeat(self.PW, len(iA), 0)
            else:
                r = rg.uniform(0, nf - 1, len(iA)); i0 = np.minimum(r.astype(int), nf - 2); t = (r - i0)[:, None]
                C = self.PW[i0] * (1 - t) + self.PW[i0 + 1] * t
            W[iA] = self._reflect(C + rg.standard_normal(C.shape) * self.jit)
        if len(iB): W[iB] = rg.uniform(self.env_lo, self.env_hi, (len(iB), NP))     # mode 1: widened envelope
        W[:, self.j_ve] = rg.uniform(self.lo_w[self.j_ve], self.hi_w[self.j_ve], n)  # electron speed: always full range
        TH = W.copy(); TH[:, self.islog] = 10 ** W[:, self.islog]
        return TH, mode

    def sample_z(self, n, rg):
        cols = []
        for zs, nom in ((self.cx.ne_z, self.cx.cfg["z_nominal"][0]), (self.cx.xe_z, self.cx.cfg["z_nominal"][1])):
            idx = rg.integers(0, len(zs), n)
            if nom in zs: idx[rg.random(n) < self.cx.cfg["z_nominal_boost"]] = zs.index(nom)
            cols.append(idx)
        return np.stack(cols, 1).astype(np.int16)

    def sample_instr(self, n, rg):
        a = np.log10(self.cx.cfg["instr_spread"]); cen = np.log10(self.cx.cfg["instr_center"])
        return 10 ** (cen + rg.uniform(-a, a, (n, 2)))


# ------------------------------------------------------------------------------------------------ data generation
def _gen_chunk(cfg, TH, ZI, INS):
    cx = get_ctx(cfg); out = np.zeros((len(TH), cx.n_wl), np.float32); ok = np.zeros(len(TH), bool)
    for i in range(len(TH)):
        try: s = cx.forward(TH[i], int(ZI[i, 0]), int(ZI[i, 1]), INS[i, 0], INS[i, 1])
        except Exception: continue
        if np.all(np.isfinite(s)): out[i], ok[i] = s, True
    return out, ok


def make_shard(cx, sp, n, seed, path, chunk=500):
    rg = np.random.default_rng(seed); parts = []; need = n
    while need > 0:
        th, mode = sp.sample(need, rg); zi = sp.sample_z(need, rg); ins = sp.sample_instr(need, rg)
        res = Parallel(n_jobs=N_JOBS)(delayed(_gen_chunk)(cx.cfg, th[i:i + chunk], zi[i:i + chunk], ins[i:i + chunk])
                                      for i in range(0, need, chunk))
        x = np.concatenate([r[0] for r in res]); ok = np.concatenate([r[1] for r in res])
        parts.append((x[ok], th[ok], zi[ok], ins[ok], mode[ok])); need -= int(ok.sum())
    cat = lambda j: np.concatenate([p[j] for p in parts])
    tmp = Path(path).with_name(Path(path).stem + ".tmp.npz")
    np.savez(tmp, X=cat(0), TH=cat(1), ZI=cat(2), INS=cat(3), MODE=cat(4)); os.replace(tmp, path)


def data_dir_for(cfg): return Path(OUT_DIR) / f"data_{cfg['tag']}"


def selftest(cx, verbose=True):
    """forward() must reproduce plasmapy's lmfit model (for several Z) - same check as in your notebook."""
    rg = np.random.default_rng(1); worst = 0.0
    for t in range(3):
        zi, zj = int(rg.integers(len(cx.ne_z))), int(rg.integers(len(cx.xe_z)))
        th = np.array([4e24, 60., 400., 20., 3e4, -2e4, 1e4, 5e3, 0.3]); sg, gm = rg.uniform(5e-12, 2e-11, 2)
        P = build_params(cx, init=dict(zip(PARAM_NAMES, th)))
        mdl = thomson.spectral_density_model(cx.wl_m, cx.settings(zi, zj, sg, gm), P)
        ref = mdl.eval(params=P, wavelengths=cx.wl_m); worst = max(worst, np.abs(cx.forward(th, zi, zj, sg, gm) - ref).max())
    assert worst < 1e-6, f"forward() differs from plasmapy's model by {worst:g}"
    t0 = time.time(); [cx.forward(th, 0, 0, 1e-11, 1e-11) for _ in range(100)]; ms = (time.time() - t0) / 100 * 1e3
    if verbose: print(f"selftest OK: max |forward - spectral_density_model| = {worst:.2e};  {ms:.2f} ms / spectrum / core")
    return ms


def cmd_generate(cfg):
    cx = get_ctx(cfg); sp = Sampler(cx); dd = data_dir_for(cfg); dd.mkdir(parents=True, exist_ok=True)
    ms = selftest(cx)
    (Path(OUT_DIR) / "meta.json").write_text(json.dumps(cfg, indent=1, default=str))
    ns = int(np.ceil(cfg["n_spectra"] / cfg["shard_size"]))
    jobs = [("val.npz", cfg["n_val"], cfg["seed"] + 1), ("test.npz", cfg["n_test"], cfg["seed"] + 2)] + \
           [(f"shard_{i:05d}.npz", cfg["shard_size"], cfg["seed"] + 1000 + i) for i in range(ns)]
    total = sum(j[1] for j in jobs); gb = total * cx.n_wl * 4 / 1e9
    ncpu = os.cpu_count() or 1; ncpu = ncpu if N_JOBS in (-1, None) else min(ncpu, N_JOBS)
    print(f"data set: {cfg['n_spectra']:,} train + {cfg['n_val']:,} val + {cfg['n_test']:,} test spectra x {cx.n_wl} pixels "
          f"(~{gb:.1f} GB) in {dd}\n  estimated generation time ~{total * ms / 1e3 / ncpu / 60:.0f} min on {ncpu} core(s)")
    free = shutil.disk_usage(dd).free / 1e9
    if free < gb * 1.1: print(f"  WARNING: only {free:.1f} GB free on this disk")
    t0 = time.time(); done = 0
    for name, n, seed in jobs:
        path = dd / name
        if path.exists(): done += n; continue
        make_shard(cx, sp, n, seed, path); done += n
        el = time.time() - t0; print(f"  {name}: {done:,}/{total:,} spectra  [{el / 60:.1f} min elapsed]", flush=True)
    print("data generation complete")


# ------------------------------------------------------------------------------------------------ CNN
for _g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(_g, True)
    except Exception: pass


@keras.utils.register_keras_serializable(package="ts")
class RandomWhiteNoise(layers.Layer):
    """Training-time noise, amplitude ~U(0, max_frac) * spectrum max, redrawn for every spectrum in every batch."""
    def __init__(self, max_frac=0.05, **kw): super().__init__(**kw); self.max_frac = max_frac
    def call(self, x, training=None):
        if not training: return x
        amp = tf.random.uniform([tf.shape(x)[0], 1], 0.0, self.max_frac) * tf.reduce_max(x, axis=1, keepdims=True)
        return x + amp * tf.random.normal(tf.shape(x))
    def get_config(self): return {**super().get_config(), "max_frac": self.max_frac}


@keras.utils.register_keras_serializable(package="ts")
class DistributionParams(layers.Layer):
    """[normal head (2*NN_), beta head (2), charge-state logits] -> [mu(NN_), sigma(NN_), alpha*, beta*, logits...]"""
    def __init__(self, sig_eps=2e-3, beta_eps=1e-3, **kw): super().__init__(**kw); self.sig_eps, self.beta_eps = sig_eps, beta_eps
    def call(self, inputs):
        hn, hb, hz = inputs; k = hn.shape[-1] // 2
        return tf.concat([tf.nn.softplus(hn[:, :k]), tf.nn.softplus(hn[:, k:]) + self.sig_eps,
                          tf.nn.softplus(hb) + self.beta_eps, hz], -1)
    def compute_output_shape(self, s): return (s[0][0], s[0][-1] + s[1][-1] + s[2][-1])
    def get_config(self): return {**super().get_config(), "sig_eps": self.sig_eps, "beta_eps": self.beta_eps}


@keras.utils.register_keras_serializable(package="ts")
class BiasInit(keras.initializers.Initializer):
    def __init__(self, values): self.values = [float(v) for v in values]
    def __call__(self, shape, dtype=None): return tf.constant(self.values, dtype=dtype or "float32")
    def get_config(self): return {"values": self.values}


ARCHS = {"light":  dict(blocks=((11, 32, 2, 4), (7, 48, 2, 4), (5, 64, 2, 2)), tail=(), dense=(128, 64)),
         "medium": dict(blocks=((11, 48, 2, 4), (7, 64, 2, 4), (5, 96, 2, 2)), tail=(), dense=(256, 128)),
         "paper":  dict(blocks=((11, 104, 6, 8), (8, 108, 6, 8)), tail=((6, 108), (6, 108), (4, 108), (4, 108), (1, 16)),
                        dense=(150, 50))}


def build_cnn(cx):
    cfg = cx.cfg; a = ARCHS[cfg["arch"]]; k_z = len(cx.ne_z) + len(cx.xe_z)
    spec = keras.Input((cx.n_wl,), name="spectrum"); ins = keras.Input((2,), name="instr")
    x = layers.Reshape((cx.n_wl, 1))(RandomWhiteNoise(cfg["noise_max_frac"])(spec))
    def conv(x, f, k): return layers.ELU()(layers.BatchNormalization()(layers.Conv1D(f, k, padding="same")(x)))
    for k, f, n, pool in a["blocks"]:
        for _ in range(n): x = conv(x, f, k)
        x = layers.MaxPooling1D(pool)(x)
    for k, f in a["tail"]: x = conv(x, f, k)
    x = layers.Concatenate()([layers.Flatten()(x), layers.Dense(16, activation="elu")(ins)])    # instrument widths join here
    for h in a["dense"]: x = layers.Dense(h, activation="elu")(x)
    hn = layers.Dense(2 * NN_, bias_initializer=BiasInit([-0.43] * NN_ + [-1.0] * NN_), name="head_normal")(x)
    hb = layers.Dense(2, bias_initializer=BiasInit([0.5, 0.5]), name="head_beta")(x)
    hz = layers.Dense(k_z, name="head_charge")(x)
    return keras.Model({"spectrum": spec, "instr": ins}, DistributionParams(name="dist_params")([hn, hb, hz]))


def make_loss(k_ne, k_xe, z_weight):
    c = 0.5 * np.log(2 * np.pi)
    def ts_loss(y_true, p):
        y = tf.cast(y_true, p.dtype); u_true, zc = y[:, :NP], tf.cast(y[:, NP:NP + 2], tf.int32)
        mu, sig, a, b = p[:, :NN_], p[:, NN_:2 * NN_], p[:, 2 * NN_], p[:, 2 * NN_ + 1]; lg = p[:, 2 * NN_ + 2:]
        nll_n = tf.reduce_sum(c + tf.math.log(sig) + 0.5 * tf.square((u_true[:, :NN_] - mu) / sig), -1)
        yb = tf.clip_by_value(u_true[:, NN_], 5e-6, 0.995)
        nll_b = -(tf.math.lgamma(a + b) - tf.math.lgamma(a) - tf.math.lgamma(b) + (a - 1) * tf.math.log(yb) + (b - 1) * tf.math.log1p(-yb))
        mae = tf.reduce_mean(tf.abs(u_true - tf.concat([mu, (a / (a + b))[:, None]], -1)), -1)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=zc[:, 0], logits=lg[:, :k_ne]) + \
             tf.nn.sparse_softmax_cross_entropy_with_logits(labels=zc[:, 1], logits=lg[:, k_ne:k_ne + k_xe])
        return tf.reduce_mean(nll_n + nll_b + mae) + z_weight * tf.reduce_mean(ce)
    return ts_loss


# ------------------------------------------------------------------------------------------------ data feeding
def load_shard(cx, path):
    d = np.load(path)
    Y = np.concatenate([cx.to_unit(d["TH"]), d["ZI"]], 1).astype(np.float32)      # 9 scaled params + 2 charge-state indices
    return d["X"], cx.scale_instr(d["INS"][:, 0], d["INS"][:, 1]), Y, d["MODE"], d["TH"], d["INS"]


def add_noise(X, rg, max_frac):
    amp = rg.uniform(0, max_frac, (len(X), 1)) * X.max(1, keepdims=True)
    return (X + amp * rg.standard_normal(X.shape)).astype(np.float32)


def train_dataset(cx, files, batch, seed):
    """Endless stream over the shards on disk (shard order and in-shard order reshuffled every pass)."""
    def gen():
        rg = np.random.default_rng(seed)
        while True:
            for si in rg.permutation(len(files)):
                X, I, Y, *_ = load_shard(cx, files[si]); p = rg.permutation(len(X))
                for s in range(0, len(X) - batch + 1, batch):
                    idx = p[s:s + batch]; yield {"spectrum": X[idx], "instr": I[idx]}, Y[idx]
    sig = ({"spectrum": tf.TensorSpec((batch, cx.n_wl), tf.float32), "instr": tf.TensorSpec((batch, 2), tf.float32)},
           tf.TensorSpec((batch, NP + 2), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)


class TimeLimit(keras.callbacks.Callback):
    def __init__(self, minutes): super().__init__(); self.minutes = minutes
    def on_train_begin(self, logs=None): self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        el = (time.time() - self.t0) / 60; print(f"   [{el:.1f} min elapsed]")
        if self.minutes and el > self.minutes: print("   time budget reached -> stopping"); self.model.stop_training = True


class LivePlots(keras.callbacks.Callback):
    """After each epoch: score the model on a fixed noisy validation subset and redraw 4 PNGs in <OUT_DIR>/live/
       01_loss.png, 02_metrics.png (R2 / coverage68 / charge-state accuracy per epoch), 03_pred_vs_true.png, 04_charge_state.png,
       05_spectra.png (simulated spectra + CNN fit; a copy of the PNG and a CSV are also kept per epoch in live/epochs/)"""
    def __init__(self, cx, X, INS, Y, TH, out, every=1, Xc=None, n_spec=6):
        super().__init__(); self.cx, self.X, self.INS, self.Y, self.TH, self.every, self.Xc = cx, X, INS, Y, TH, every, Xc
        self.dir = Path(out) / "live"; (self.dir / "epochs").mkdir(parents=True, exist_ok=True)
        self.spec_idx = np.random.default_rng(123).choice(len(X), min(n_spec, len(X)), replace=False)   # same spectra every epoch
        self.h = {k: [] for k in ("epoch", "loss", "val_loss", "lr", "r2", "cov", "zacc", "zpm1")}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.every: return
        try: self._update(epoch, logs)
        except Exception as e: print(f"   (live plots skipped: {e})")

    def _update(self, epoch, logs):
        cx, h, U = self.cx, self.h, self.Y[:, :NP]; zt = self.Y[:, NP:NP + 2].astype(int)
        pr = predict(self.model, cx, self.X, self.INS[:, 0], self.INS[:, 1])
        r2 = [1 - np.sum((U[:, j] - pr["med_u"][:, j]) ** 2) / np.sum((U[:, j] - U[:, j].mean()) ** 2) for j in range(NP)]
        cov = [np.mean((U[:, j] >= pr["lo68_u"][:, j]) & (U[:, j] <= pr["hi68_u"][:, j])) for j in range(NP)]
        za = [np.mean(pr[k].argmax(1) == zt[:, c]) for k, c in (("pz_ne", 0), ("pz_xe", 1))]
        zp = [np.mean(np.abs(pr[k].argmax(1) - zt[:, c]) <= 1) for k, c in (("pz_ne", 0), ("pz_xe", 1))]
        lr = logs.get("learning_rate", logs.get("lr", np.nan))
        for k, v in (("epoch", epoch + 1), ("loss", logs.get("loss", np.nan)), ("val_loss", logs.get("val_loss", np.nan)),
                     ("lr", lr), ("r2", r2), ("cov", cov), ("zacc", za), ("zpm1", zp)): h[k].append(v)
        print(f"   [live] mean R2 {np.mean(r2):.3f} | coverage68 {np.mean(cov):.2f} | Z_Ne acc {za[0]:.2f}, Z_Xe acc {za[1]:.2f}")
        ep = np.array(h["epoch"])
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        axs[0].plot(ep, h["loss"], "o-", label="train (noisy batches)"); axs[0].plot(ep, h["val_loss"], "o-", label="validation")
        axs[0].set_yscale("symlog"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss"); axs[0].legend(); axs[0].grid(alpha=.3)
        axs[1].semilogy(ep, h["lr"], "o-"); axs[1].set_xlabel("epoch"); axs[1].set_ylabel("learning rate"); axs[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(self.dir / "01_loss.png", dpi=110); plt.close(fig)
        R, C = np.array(h["r2"]), np.array(h["cov"]); Z, Z1 = np.array(h["zacc"]), np.array(h["zpm1"])
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for j, k in enumerate(PARAM_NAMES): axs[0].plot(ep, R[:, j], "o-", ms=3, label=k)
        axs[0].set_ylim(-0.1, 1.02); axs[0].set_title("R2 of predicted median (scaled space)"); axs[0].legend(fontsize=7, ncol=2)
        for j, k in enumerate(PARAM_NAMES): axs[1].plot(ep, C[:, j], "o-", ms=3)
        axs[1].axhline(0.68, color="k", ls="--"); axs[1].set_ylim(0, 1); axs[1].set_title("coverage of the 68 % interval (target 0.68)")
        for i, nm in enumerate(("Ne", "Xe")):
            l, = axs[2].plot(ep, Z[:, i], "o-", label=f"Z_{nm} exact"); axs[2].plot(ep, Z1[:, i], "s--", color=l.get_color(), label=f"Z_{nm} within +-1")
        axs[2].set_ylim(0, 1); axs[2].set_title("charge-state accuracy"); axs[2].legend(fontsize=8)
        for a in axs: a.set_xlabel("epoch"); a.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(self.dir / "02_metrics.png", dpi=110); plt.close(fig)
        fig, axs = plt.subplots(3, 3, figsize=(11, 10))
        for ax, (j, k) in zip(axs.ravel(), enumerate(PARAM_NAMES)):
            ax.scatter(self.TH[:, j], pr["med"][:, j], s=3, alpha=.3, rasterized=True)
            lim = [self.TH[:, j].min(), self.TH[:, j].max()]; ax.plot(lim, lim, "k--", lw=.8); ax.set_title(f"{k}  (R2={r2[j]:.2f})", fontsize=9)
            if k in cx.log: ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("true"); ax.set_ylabel("predicted median")
        fig.suptitle(f"validation subset after epoch {epoch + 1}"); plt.tight_layout(); plt.savefig(self.dir / "03_pred_vs_true.png", dpi=100); plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (nm, k, c, zs) in zip(axs, (("Ne", "pz_ne", 0, cx.ne_z), ("Xe", "pz_xe", 1, cx.xe_z))):
            M = np.zeros((len(zs), len(zs))); np.add.at(M, (zt[:, c], pr[k].argmax(1)), 1); M /= np.maximum(M.sum(1, keepdims=True), 1)
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis"); ax.set_xticks(range(len(zs))); ax.set_xticklabels(zs); ax.set_yticks(range(len(zs)))
            ax.set_yticklabels(zs); ax.set_xlabel("predicted Z"); ax.set_ylabel("true Z"); ax.set_title(f"{nm} charge state"); plt.colorbar(im, ax=ax)
        plt.tight_layout(); plt.savefig(self.dir / "04_charge_state.png", dpi=110); plt.close(fig)
        n = len(self.spec_idx); nc = min(3, n); nr = int(np.ceil(n / nc)); fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 3.2 * nr), squeeze=False)
        cols = {"wavelength_nm": cx.wl_nm}
        for ax, i in zip(axs.ravel(), self.spec_idx):
            zi, zj = int(pr["pz_ne"][i].argmax()), int(pr["pz_xe"][i].argmax())
            try: fit = cx.forward(pr["med"][i], zi, zj, self.INS[i, 0], self.INS[i, 1])
            except Exception: fit = np.full(cx.n_wl, np.nan)
            r2s = 1 - np.var(self.X[i] - fit) / np.var(self.X[i]); clean = self.Xc[i] if self.Xc is not None else self.X[i]
            ax.plot(cx.wl_nm, self.X[i], color="0.75", lw=.7, label="noisy input"); ax.plot(cx.wl_nm, clean, "k", lw=1, label="true (simulated)")
            ax.plot(cx.wl_nm, fit, "r", lw=1, label="CNN fit")
            ax.set_title(f"#{i}  R2={r2s:.3f}  Ne {cx.ne_z[zt[i, 0]]}+/{cx.ne_z[zi]}+  Xe {cx.xe_z[zt[i, 1]]}+/{cx.xe_z[zj]}+ (true/pred)", fontsize=8)
            ax.set_xlabel("nm"); cols[f"data_{i}"], cols[f"true_{i}"], cols[f"fit_{i}"] = self.X[i], clean, fit
        for ax in axs.ravel()[n:]: ax.axis("off")
        axs.ravel()[0].legend(fontsize=7); fig.suptitle(f"simulated validation spectra after epoch {epoch + 1}"); plt.tight_layout()
        plt.savefig(self.dir / "05_spectra.png", dpi=110); plt.savefig(self.dir / "epochs" / f"spectra_epoch_{epoch + 1:03d}.png", dpi=90); plt.close(fig)
        pd.DataFrame(cols).to_csv(self.dir / "epochs" / f"spectra_epoch_{epoch + 1:03d}.csv", index=False)


def load_model(path): return keras.models.load_model(path, compile=False)


def cmd_train(cfg, resume=False):
    cx = get_ctx(cfg); out = Path(OUT_DIR); dd = data_dir_for(cfg)
    files = sorted(dd.glob("shard_*.npz"))
    if not files or not (dd / "val.npz").exists(): raise SystemExit(f"no data in {dd}: run 'generate' first")
    (out / "meta.json").write_text(json.dumps(cfg, indent=1, default=str))
    keras.utils.set_random_seed(cfg["seed"])
    best, last = out / "ts_model.keras", out / "ts_model_last.keras"
    if resume and last.exists(): model = load_model(last); print("resuming from", last)
    else: model = build_cnn(cx)
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"], clipnorm=1.0),
                  loss=make_loss(len(cx.ne_z), len(cx.xe_z), cfg["z_loss_weight"]))
    Xv, Iv, Yv, _, THv, INSv = load_shard(cx, dd / "val.npz")
    nv = min(len(Xv), 20_000); Xc = Xv[:nv].copy(); Xv = add_noise(Xv[:nv], np.random.default_rng(cfg["seed"] + 5), cfg["noise_max_frac"])
    spe = len(files) * (cfg["shard_size"] // cfg["batch"])
    print(f"arch '{cfg['arch']}': {model.count_params():,} parameters | {len(files) * cfg['shard_size']:,} training spectra "
          f"({spe} steps/epoch) | {nv:,} validation | GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    cbs = [keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
           keras.callbacks.EarlyStopping(patience=cfg["patience"]),
           keras.callbacks.ModelCheckpoint(str(best), monitor="val_loss", save_best_only=True),
           keras.callbacks.ModelCheckpoint(str(last)),
           keras.callbacks.CSVLogger(str(out / "history.csv"), append=resume), TimeLimit(TIME_BUDGET_MIN)]
    if LIVE_PLOTS:
        m = min(LIVE_N, nv); cbs.append(LivePlots(cx, Xv[:m], INSv[:m], Yv[:m], THv[:m], out, LIVE_EVERY, Xc[:m], LIVE_SPECTRA)); print(f"live progress figures -> {out / 'live'}/")
    hist = model.fit(train_dataset(cx, files, cfg["batch"], cfg["seed"]), steps_per_epoch=spe, epochs=cfg["epochs"],
                     validation_data=({"spectrum": Xv, "instr": Iv[:nv]}, Yv[:nv]), validation_batch_size=1024,
                     callbacks=cbs, verbose=2)
    plt.figure(figsize=(5, 3)); plt.plot(hist.history["loss"], label="train (noisy)"); plt.plot(hist.history["val_loss"], label="validation")
    plt.yscale("symlog"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "train_curves.png", dpi=150); plt.close()
    print(f"best model saved -> {best}")


# ------------------------------------------------------------------------------------------------ predictions
def softmax(z): e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def unit_quantiles(p, q):
    mu, sig, a, b = p[:, :NN_], p[:, NN_:2 * NN_], p[:, 2 * NN_], p[:, 2 * NN_ + 1]; U = np.empty((len(p), NP))
    U[:, :NN_] = stats.norm.ppf(q, mu, sig); U[:, NN_] = stats.beta.ppf(q, a, b); return np.clip(U, 0, 1)


def predict(model, cx, X, sigma, gamma, bs=1024):
    """Spectra (max-normalised, on the training grid) -> medians, intervals (physical units) and charge-state probabilities."""
    p = model.predict({"spectrum": np.asarray(X, np.float32), "instr": cx.scale_instr(sigma, gamma)}, batch_size=bs, verbose=0)
    lg = p[:, 2 * NN_ + 2:]; k = len(cx.ne_z); f = lambda q: unit_quantiles(p, q)
    out = dict(raw=p, pz_ne=softmax(lg[:, :k]), pz_xe=softmax(lg[:, k:]))
    for name, q in (("med", .5), ("lo68", .158), ("hi68", .842), ("lo", .005), ("hi", .995)):
        out[name + "_u"] = f(q); out[name] = cx.from_unit(out[name + "_u"])
    return out


def cmd_evaluate(cfg, model=None):
    cx = get_ctx(cfg); out = Path(OUT_DIR); dd = data_dir_for(cfg)
    model = model or load_model(out / "ts_model.keras")
    X, I, Y, M, TH, INS = load_shard(cx, dd / "test.npz")
    Xn = add_noise(X, np.random.default_rng(cfg["seed"] + 7), cfg["noise_max_frac"])
    pr = predict(model, cx, Xn, INS[:, 0], INS[:, 1]); U = Y[:, :NP]; rows = []
    for j, k in enumerate(PARAM_NAMES):
        r2 = 1 - np.sum((U[:, j] - pr["med_u"][:, j]) ** 2) / np.sum((U[:, j] - U[:, j].mean()) ** 2)
        row = dict(param=k, R2=r2, MAE_pct_of_range=100 * np.mean(np.abs(U[:, j] - pr["med_u"][:, j])),
                   coverage68=np.mean((U[:, j] >= pr["lo68_u"][:, j]) & (U[:, j] <= pr["hi68_u"][:, j])))
        for m, nm in enumerate(MODE_NAMES):
            if (M == m).any(): row[f"MAE%_{nm}"] = 100 * np.mean(np.abs(U[M == m, j] - pr["med_u"][M == m, j]))
        rows.append(row)
    met = pd.DataFrame(rows).set_index("param"); met.round(3).to_csv(out / "eval_metrics.csv")
    print("\nHeld-out test set (noisy). R2/MAE in the CNN's scaled space (log10 for n and temperatures);\n"
          "coverage68 should be ~0.68 for a calibrated network:\n", met.round(3).to_string())
    zt = Y[:, NP:NP + 2].astype(int); zres = {}
    for nm, pz, col in (("Z_Ne", pr["pz_ne"], 0), ("Z_Xe", pr["pz_xe"], 1)):
        top1 = pz.argmax(1) == zt[:, col]; top3 = (np.argsort(-pz, 1)[:, :3] == zt[:, [col]]).any(1)
        pm1 = np.abs(pz.argmax(1) - zt[:, col]) <= 1
        zres[nm] = (top1.mean(), pm1.mean(), top3.mean())
        print(f"{nm}: top-1 accuracy {top1.mean():.3f} | within +-1 charge state {pm1.mean():.3f} | in top-3 {top3.mean():.3f}")
    pd.DataFrame(zres, index=["top1", "within_pm1", "top3"]).T.round(4).to_csv(out / "eval_charge_state.csv")

    sub = np.random.default_rng(0).choice(len(U), min(5000, len(U)), replace=False)
    fig, axs = plt.subplots(3, 3, figsize=(11, 10))
    for ax, (j, k) in zip(axs.ravel(), enumerate(PARAM_NAMES)):
        ax.scatter(TH[sub, j], pr["med"][sub, j], s=3, alpha=.25, rasterized=True)
        lim = [TH[sub, j].min(), TH[sub, j].max()]; ax.plot(lim, lim, "k--", lw=.8); ax.set_title(k)
        if k in cx.log: ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("true"); ax.set_ylabel("predicted median")
    plt.tight_layout(); plt.savefig(out / "eval_scatter.png", dpi=150); plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (nm, pz, col, zs) in zip(axs, (("Ne", pr["pz_ne"], 0, cx.ne_z), ("Xe", pr["pz_xe"], 1, cx.xe_z))):
        C = np.zeros((len(zs), len(zs))); np.add.at(C, (zt[:, col], pz.argmax(1)), 1); C /= np.maximum(C.sum(1, keepdims=True), 1)
        im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis"); ax.set_xticks(range(len(zs))); ax.set_xticklabels(zs); ax.set_yticks(range(len(zs)))
        ax.set_yticklabels(zs); ax.set_xlabel("predicted Z"); ax.set_ylabel("true Z"); ax.set_title(f"{nm} charge state"); plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(out / "eval_charge_confusion.png", dpi=150); plt.close()
    print(f"plots/tables saved in {out}/")


# ------------------------------------------------------------------------------------------------ lmfit helpers (as in your notebook)
def build_params(cx, init=None, bounds=None):
    init = {**cx.cfg["init_values"], **(init or {})}; bnd = {**cx.fb, **(bounds or {})}; r = cx.xe_to_h; P = Parameters()
    def add(name):
        lo, hi = bnd[name]; P.add(name, value=float(np.clip(init[name], lo, hi)), min=lo, max=hi, vary=True)
    add("n"); add("T_i_0"); add("T_i_1"); P.add("T_i_2", expr="T_i_1")
    add("ion_speed_0"); add("ion_speed_1"); add("ion_speed_2"); add("ifract_0")
    P.add("ifract_1", expr=f"(1 - ifract_0) * 1/(1 + {r})"); P.add("ifract_2", expr=f"(1 - ifract_0) * {r}/(1 + {r})")
    add("T_e_0"); add("electron_speed_0")
    return P


def narrowed_bounds(cx, med, lo, hi):
    """Fit bounds narrowed to the CNN's 0.5-99.5 % interval (at least MIN_WIDTH of the full range wide)."""
    bnd = {}; mw = cx.cfg["min_width"]
    for j, k in enumerate(PARAM_NAMES):
        flo, fhi = cx.fb[k]; a, b, m = lo[j], hi[j], med[j]
        if k in cx.log: w = mw * (np.log10(fhi) - np.log10(flo)); a = 10 ** min(np.log10(a), np.log10(m) - w / 2); b = 10 ** max(np.log10(b), np.log10(m) + w / 2)
        else:           w = mw * (fhi - flo);                       a = min(a, m - w / 2);                       b = max(b, m + w / 2)
        bnd[k] = (max(a, flo), min(b, fhi))
    return bnd


def fit_spectrum(cx, settings, skw, params, method, max_tries=1, max_nfev=None):
    t0 = time.time(); best = None
    for _ in range(max_tries):
        P = copy.deepcopy(params); mdl = thomson.spectral_density_model(cx.wl_m, settings, P)
        res = mdl.fit(skw, params=copy.deepcopy(P), wavelengths=cx.wl_m, method=method, max_nfev=max_nfev)
        r2 = 1 - res.residual.var() / np.var(skw)
        if best is None or r2 > best[2]: best = (res, mdl, r2)
        if r2 >= cx.cfg["r2_target"]: break
    return (*best, time.time() - t0)


def refine_one(cfg, name, skw, sigma, gamma, med, lo, hi, combos, outdir):
    """lmfit differential evolution with CNN-narrowed bounds for the most probable charge-state combinations;
    falls back to your full bounds if R2 stays below target. Writes the same kind of files as your script."""
    cx = get_ctx(cfg); tgt = cfg["r2_target"]; os.makedirs(outdir, exist_ok=True)
    init, bnd = dict(zip(PARAM_NAMES, med)), narrowed_bounds(cx, med, lo, hi); best = None; used = "CNN-narrowed bounds"
    for zi, zj in combos:
        res, mdl, r2, _ = fit_spectrum(cx, cx.settings(zi, zj, sigma, gamma), skw, build_params(cx, init, bnd),
                                       "differential_evolution", cfg["refine_tries"], cfg["narrow_max_nfev"])
        if best is None or r2 > best[2]: best = (res, mdl, r2, zi, zj)
        if r2 >= tgt: break
    if best[2] < tgt:
        zi, zj = combos[0]
        res, mdl, r2, _ = fit_spectrum(cx, cx.settings(zi, zj, sigma, gamma), skw, build_params(cx), "differential_evolution", 4)
        if r2 > best[2]: best = (res, mdl, r2, zi, zj); used = "full bounds (fallback)"
    res, mdl, r2, zi, zj = best; ions = cx.ions(zi, zj); wl = cx.wl_nm
    fig, ax = plt.subplots(); ax.plot(wl, skw, label="Data"); ax.plot(wl, res.best_fit, label="Best fit")
    ax.axvline(cfg["probe_nm"], color="red"); ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Skw"); ax.legend()
    ax.set_title(f"{name}: R2={r2:.3f}, {ions[0]}, {ions[2]}", fontsize=9)
    fig.savefig(f"{outdir}/{name}_fit.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame({"wavelength_nm": wl, "data": skw, "best_fit": res.best_fit}).to_csv(f"{outdir}/{name}_fit_spectrum.csv", index=False)
    plt.figure(); plt.plot(wl, res.best_fit, "k", lw=2, label="Total fit"); zion = np.array([cx.ne_z[zi], 1, cx.xe_z[zj]], float)
    for i in range(3):
        p = res.params.copy()
        for j in range(3): p[f"ifract_{j}"].set(value=1.0 if j == i else 0.0, expr="" if j else None, vary=False)
        w = np.array([res.params[f"ifract_{j}"].value * zion[j] for j in range(3)]); w /= w.sum()
        plt.plot(wl, mdl.eval(params=p, wavelengths=cx.wl_m) * w[i], label=ions[i])
    plt.legend(); plt.savefig(f"{outdir}/{name}_contributions.png", dpi=200, bbox_inches="tight"); plt.close()
    rows = [{"parameter": k, "value": v.value, "stderr": v.stderr} for k, v in res.params.items()]
    rows += [{"parameter": "Z_Ne", "value": cx.ne_z[zi], "stderr": np.nan}, {"parameter": "Z_Xe", "value": cx.xe_z[zj], "stderr": np.nan}]
    pd.DataFrame(rows).to_csv(f"{outdir}/{name}_params.csv", index=False)
    out = {"fit_R2": r2, "fit_accepted": bool(r2 >= tgt), "fit_bounds_used": used, "fit_Z_Ne": cx.ne_z[zi], "fit_Z_Xe": cx.xe_z[zj],
           "fit_redchi": res.redchi}
    out.update({f"fit_{k}": res.params[k].value for k in PARAM_NAMES}); return out


# ------------------------------------------------------------------------------------------------ analyse your spectra
def cmd_analyze(patterns, out_dir=None, refine=False):
    mp = Path(OUT_DIR) / "meta.json"
    if not mp.exists(): raise SystemExit(f"{mp} not found: train first (or set OUT_DIR to the folder holding the trained model)")
    cfg = json.loads(mp.read_text()); cx = get_ctx(cfg); model = load_model(Path(OUT_DIR) / "ts_model.keras")
    out_dir = Path(out_dir or Path(OUT_DIR) / "analysis"); out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted({f for pat in patterns for f in (glob.glob(pat) or [pat]) if Path(f).is_file()})
    if not paths: raise SystemExit("no input files found")
    names, X, SIG, GAM = [], [], [], []; c = cfg["instr_center"]
    for p in paths:
        try: wl, y, s, g = load_ts_csv(p)
        except Exception as e: print(f"skipping {p}: {e}"); continue
        if wl[0] > cx.wl_nm[0] + 0.01 or wl[-1] < cx.wl_nm[-1] - 0.01 or y.max() <= 0:
            print(f"skipping {p}: needs positive data covering {cx.wl_nm[0]:.3f}-{cx.wl_nm[-1]:.3f} nm (file: {wl[0]:.3f}-{wl[-1]:.3f})"); continue
        yg = np.interp(cx.wl_nm, wl, y); names.append(Path(p).stem); X.append(yg / yg.max()); SIG.append(s or c[0]); GAM.append(g or c[1])
    X, SIG, GAM = np.array(X), np.array(SIG), np.array(GAM); sc = cx.scale_instr(SIG, GAM)
    for nm, v in zip(names, sc):
        if v.min() < -0.25 or v.max() > 1.25: print(f"  WARNING {nm}: sigma/gamma outside the instrument range seen in training -> less reliable")
    pr = predict(model, cx, X, SIG, GAM); rows = []
    joint = pr["pz_ne"][:, :, None] * pr["pz_xe"][:, None, :]; K = len(cx.xe_z)
    for i, nm in enumerate(names):
        top = np.argsort(-joint[i].ravel())[:max(3, cfg["refine_top_z"])]; combos = [(int(t // K), int(t % K)) for t in top]
        zi, zj = combos[0]; cnn = cx.forward(pr["med"][i], zi, zj, SIG[i], GAM[i])
        r = {"file": nm, "sigma_m": SIG[i], "gamma_m": GAM[i], "Z_Ne": cx.ne_z[zi], "P_Z_Ne": pr["pz_ne"][i].max(),
             "Z_Xe": cx.xe_z[zj], "P_Z_Xe": pr["pz_xe"][i].max(),
             "top_Z_combos": "; ".join(f"Ne{cx.ne_z[a]}+/Xe{cx.xe_z[b]}+ ({joint[i, a, b]:.2f})" for a, b in combos[:3]),
             "cnn_R2": 1 - np.var(X[i] - cnn) / np.var(X[i])}
        for j, k in enumerate(PARAM_NAMES): r[k] = pr["med"][i, j]; r[k + "_lo68"] = pr["lo68"][i, j]; r[k + "_hi68"] = pr["hi68"][i, j]
        rows.append(r)
        fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(cx.wl_nm, X[i], label="data"); ax.plot(cx.wl_nm, cnn, label="CNN median")
        ax.axvline(cfg["probe_nm"], color="red", lw=.6); ax.set_xlabel("nm"); ax.legend(); ax.set_title(f"{nm}: Ne{cx.ne_z[zi]}+  Xe{cx.xe_z[zj]}+", fontsize=9)
        plt.tight_layout(); plt.savefig(out_dir / f"{nm}_cnn.png", dpi=150); plt.close(fig)
        if len(names) <= 5:
            print(f"\n=== {nm}   Ne {cx.ne_z[zi]}+ (p={r['P_Z_Ne']:.2f}), Xe {cx.xe_z[zj]}+ (p={r['P_Z_Xe']:.2f}),  CNN R2={r['cnn_R2']:.3f}")
            for k in PARAM_NAMES:
                s_, un = UNIT_FMT[k]; print(f"   {k:<17s} {r[k] * s_:10.4g}  [{r[k + '_lo68'] * s_:.4g}, {r[k + '_hi68'] * s_:.4g}] {un}")
    df = pd.DataFrame(rows)
    if refine:
        print(f"\nrefining {len(df)} spectra with lmfit differential evolution (CNN-narrowed bounds)...")
        jobs = [delayed(refine_one)(cfg, nm, X[i], SIG[i], GAM[i], pr["med"][i], pr["lo"][i], pr["hi"][i],
                                    [(int(t // K), int(t % K)) for t in np.argsort(-joint[i].ravel())[:cfg["refine_top_z"]]], str(out_dir / "fits"))
                for i, nm in enumerate(names)]
        df = pd.concat([df, pd.DataFrame(Parallel(n_jobs=N_JOBS)(jobs))], axis=1)
        print(f"{int(df['fit_accepted'].sum())}/{len(df)} fits reached R2 >= {cfg['r2_target']}")
    df.to_csv(out_dir / "analysis_summary.csv", index=False)
    print(f"\n{len(df)} spectra analysed -> {out_dir}/analysis_summary.csv (+ plots{', fits' if refine else ''})")
    return df


# ------------------------------------------------------------------------------------------------ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "generate", "train", "evaluate", "analyze", "selftest"])
    ap.add_argument("files", nargs="*", help="analyze: spectrum CSV files or glob patterns (quote the patterns)")
    ap.add_argument("--refine", action="store_true", help="analyze: also run the lmfit differential-evolution fit")
    ap.add_argument("--resume", action="store_true", help="train: continue from the last checkpoint")
    ap.add_argument("--out", default=None, help="analyze: output folder (default <OUT_DIR>/analysis)")
    a = ap.parse_args(); Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    if a.cmd == "analyze": cmd_analyze(a.files, a.out, a.refine); return
    cfg = make_cfg()
    if a.cmd == "selftest": selftest(get_ctx(cfg)); return
    if a.cmd in ("all", "generate"): cmd_generate(cfg)
    if a.cmd in ("all", "train"): cmd_train(cfg, a.resume)
    if a.cmd in ("all", "evaluate"): cmd_evaluate(cfg)
    if a.cmd == "all": print(f"\nDONE. Analyse your data with:\n  python {Path(sys.argv[0]).name} analyze \"your_spectra/*.csv\" [--refine]")


if __name__ == "__main__":
    main()
