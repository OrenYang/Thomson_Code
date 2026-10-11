#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ts_train_big_zlookup.py  -  train ONE large CNN for Thomson-scattering (ion-acoustic) spectra of Ne / H / Xe plasmas.

CHANGES w.r.t. ts_train_big.py
  1. Ionization is no longer random / predicted.  Z_Ne and Z_Xe are looked up from FLYCHK tables as a function of
     (Te, ne), exactly like get_ion_species() in your fitting script (round(Z_avg), min 1).  H is always H+.
     The CNN therefore has no charge-state head any more; Z follows from the predicted Te and n.
     A Gaussian jitter on Z_avg (Z_SCALE_JITTER, default 10 % = your MC dispersion) keeps the CNN honest about
     FLYCHK uncertainty.
  2. Density is drawn from its own prior (focus 1e18-1e19 cm^-3, tails to 1e17 / 1e20) instead of from the fitted
     profile, whose density was not a measurement.  FIT_BOUNDS["n"] is now 1e23-1e26 m^-3.
  3. Constraint  |v_Xe| <= |v_H|  is built in by construction: the CNN predicts f_Xe_H = v_Xe / v_H in [-1, 1]
     (set XE_H_RATIO_BOUNDS = (0, 1) to also force Xe to move in the same direction as H).
  4. Electron drift can be predicted as the drift RELATIVE to H (REL_ELECTRON = True): v_rel = v_e - v_H, and v_e is
     backed out afterwards.  (The spectrum only depends on velocity differences plus one overall Doppler shift.)
  5. The refinement fit (--refine) uses the fast forward model with Z updated on every iteration (as in your script),
     fits log10(n), and enforces v_Xe = f_Xe_H * v_H (|f| <= 1) and v_e = v_H + v_rel as real lmfit expression constraints.

Z LOOKUP FILES (one per element, plain text, read with np.loadtxt - same layout as your zforXe.txt):
        <ignored>   ne_1   ne_2  ...  ne_M          <- row 0: electron densities in cm^-3 (first cell is a dummy)
        Te_1        Z11    Z12   ...  Z1M           <- column 0: Te in eV;  body: mean charge Zbar(Te, ne)
        Te_2        Z21    ...
  needed:  zforNe.txt  and  zforXe.txt   (H is fixed to Z = 1)
  coverage: Te  0.5 - 1000 eV (log-spaced, >=10 points/decade),  ne  1e16 - 1e21 cm^-3 (log-spaced, >=3 points/decade)
  Te and ne must be strictly increasing/unique; the values are the average ionization Zbar of a pure-element plasma
  (use the same FLYCHK settings as zforXe.txt).  Outside the table range the lookup is clamped, not extrapolated.
  (No Ne table yet?  Set Z_OVERRIDE = {"Ne": 9, "Xe": None} to fix Ne at 9+ for the time being.)

USAGE
    python ts_train_big_zlookup.py all                  # generate data + train + evaluate (default)
    python ts_train_big_zlookup.py generate | train | evaluate  # individual steps (generate is resumable)
    python ts_train_big_zlookup.py train --resume
    python ts_train_big_zlookup.py analyze "spectra/*.csv" [--refine]
    python ts_train_big_zlookup.py selftest

Spectrum CSV columns: wavelength_nm, intensity  (+ optional sigma, gamma in metres).
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
from scipy.interpolate import RegularGridInterpolator
from lmfit import Parameters, Model as LmModel
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
OUT_DIR       = "ts_big_zlookup"         # everything is written here (data shards, model, plots, meta.json)
PROFILE_CSV   = "out5_averaged.csv"      # fitted profile: used to centre Te, Ti, ion speeds, ifract (NOT density)

# ---- wavelength grid / instrument (see GRID_FROM_CSV: grid, sigma, gamma are then taken from a real spectrum)
GRID_FROM_CSV = ""
WL_MIN_NM, WL_MAX_NM, N_PIX = 524.0, 529.0, 1024
INSTR_CENTER  = (1.0e-11, 1.0e-11)       # Voigt (sigma, gamma) in METRES
INSTR_SPREAD  = 2.0                      # training draws sigma, gamma log-uniformly in [centre/spread, centre*spread]
PROBE_NM, SCATTER_DEG = 526.5, 90.0

# ---- ionization: Z from FLYCHK tables (see docstring for the file format)
Z_TABLES       = {"Ne": "zforNe.txt", "Xe": "zforXe.txt"}
Z_OVERRIDE     = {"Ne": None, "Xe": None}   # int -> fixed charge state for that element (like ION_SPECIES_OVERRIDE)
Z_SCALE_JITTER = 0.0                   # Z_used = round(Zbar * (1 + N(0, this))) in the training data; 0 = exact lookup
XE_TO_H        = 0.0065                  # ifract_Xe / ifract_H, fixed as in your fits

# ---- fit bounds ('n' in m^-3, T in eV, speeds in m/s).  Used for CNN target scaling and the refinement fit.
FIT_BOUNDS = {"n": (1e23, 1e27), "T_e_0": (0.5, 800.), "T_i_0": (10., 5000.), "T_i_1": (1., 500.),
              "ion_speed_0": (-5e5, 5e5), "ion_speed_1": (-5e5, 5e5), "ion_speed_2": (-5e5, 5e5),
              "electron_speed_0": (-5e5, 5e5), "ifract_0": (0., 0.6)}

# ---- density prior (cm^-3).  No independent measurement exists, so n is drawn from this prior in log10.
DENSITY_CORE_CM3      = (1e18, 1e20)     # focus range
DENSITY_TAIL_CM3      = (1e17, 1e21)     # occasional wider draws
DENSITY_CORE_FRACTION = 0.80             # P(core draw); the rest is log-uniform over the tail range
                                         # -> ~45 % of samples in 1e18-1e19, ~45 % in 1e19-1e20, ~5 % in each outer decade

# ---- velocities
XE_H_RATIO_BOUNDS = (-1.0, 1.0)          # f_Xe_H = v_Xe / v_H  -> |v_Xe| <= |v_H| always.  (0, 1) = also same direction as H.
                                         # NOTE: must stay within [-1, 1]. When v_H ~ 0, f is unconstrained -> wide interval (harmless).
REL_ELECTRON  = True                     # True: CNN predicts v_rel = v_e - v_H and v_e = v_H + v_rel is backed out
V_REL_BOUNDS  = (-5e5, 5e5)              # prior range of v_rel (electron speed is always drawn uniformly in it).
                                         # Narrow it if you know the physics (e.g. from current / (e n)): helps v_e.
INIT_VALUES = {"n": 5e24, "T_e_0": 50., "T_i_0": 150., "T_i_1": 150., "ion_speed_0": 0., "ion_speed_1": 0.,
               "f_Xe_H": 0.5, "v_rel_e_H": 0., "electron_speed_0": 0., "ifract_0": 0.4}
LOG_PARAMS  = ["n", "T_e_0", "T_i_0", "T_i_1"]            # CNN output is scaled in log10 for these (linear for the rest)
SAMPLE_LOG  = LOG_PARAMS + ["ifract_0"]                   # drawn log-uniformly / log-normally
LOG_FLOOR   = 1e-3

# ---- how the other training parameters are drawn around the profile
MIX          = (0.60, 0.30, 0.10)       # profile-anchored / widened envelope / full fit bounds
JITTER_FLOOR = {"n": 0.35, "T_e_0": 0.20, "T_i_0": 0.20, "T_i_1": 0.30,        # decades for log params,
                "ion_speed_0": 5e4, "ion_speed_1": 6e4, "f_Xe_H": 0.25,        # m/s for speeds (f_Xe_H: ratio)
                "electron_speed_0": 0.0, "v_rel_e_H": 0.0, "ifract_0": 0.15}
ENV_PAD_DEC  = 0.40
ENV_PAD_FRAC = 0.40

# ---- data set size
N_SPECTRA  = 1_000_000
N_VAL, N_TEST = 20_000, 20_000
SHARD_SIZE = 51_200
N_JOBS     = -1
SEED       = 42

# ---- training
ARCH            = "medium"              # "light" | "medium" | "paper"
EPOCHS          = 60
BATCH, LR       = 512, 5e-4
NOISE_MAX_FRAC  = 0.05
PATIENCE        = 8
TIME_BUDGET_MIN = None
LIVE_PLOTS      = True
LIVE_EVERY, LIVE_N = 1, 3000
LIVE_SPECTRA     = 6

# ---- analysis / refinement fit
R2_TARGET, NARROW_MAX_NFEV, MIN_WIDTH = 0.98, 6000, 0.10
REFINE_TRIES = 2
# ============================================================================================================

for _k in ("OUT_DIR", "N_SPECTRA", "N_VAL", "N_TEST", "SHARD_SIZE", "BATCH", "EPOCHS", "N_JOBS", "ARCH", "PROFILE_CSV"):
    if "TS_" + _k in os.environ: globals()[_k] = type(globals()[_k])(os.environ["TS_" + _k])
if os.environ.get("TS_BUDGET"): TIME_BUDGET_MIN = float(os.environ["TS_BUDGET"])

EV_TO_K = 11604.51812155
PHYS_NAMES = ["n", "T_e_0", "T_i_0", "T_i_1", "ion_speed_0", "ion_speed_1", "ion_speed_2", "electron_speed_0", "ifract_0"]
NP, NN_ = 9, 8                          # 9 CNN targets: 8 normal heads + ifract_0 (beta head, must be last)


def net_names(rel):
    """CNN target names. Position 6 is f_Xe_H = v_Xe / v_H (|f|<=1); position 7 is v_rel = v_e - v_H or v_e itself."""
    return PHYS_NAMES[:6] + ["f_Xe_H", "v_rel_e_H" if rel else "electron_speed_0"] + PHYS_NAMES[8:]


_KMS = (1e-3, "km/s")
UNIT_FMT = {"n": (1e-24, "1e18 cm^-3"), "T_e_0": (1, "eV"), "T_i_0": (1, "eV"), "T_i_1": (1, "eV"),
            "ion_speed_0": _KMS, "ion_speed_1": _KMS, "ion_speed_2": _KMS, "electron_speed_0": _KMS,
            "f_Xe_H": (1, "= v_Xe/v_H"), "v_rel_e_H": _KMS, "ifract_0": (1, "")}
MODE_NAMES = ["profile", "envelope", "wide"]

Z_TABLE_HELP = """
  Z lookup table missing: {path}
  Needed: one text file per element (zforNe.txt, zforXe.txt), np.loadtxt layout:
      row 0   : [dummy, ne_1 ... ne_M]   electron density in cm^-3, strictly increasing
      col 0   : [Te_1 ... Te_K]          electron temperature in eV, strictly increasing
      body    : mean ionization Zbar(Te, ne) of the pure element (same FLYCHK settings as zforXe.txt)
  Recommended coverage: Te 0.5-1000 eV, ne 1e16-1e21 cm^-3, log-spaced.
  Quick workaround: set Z_OVERRIDE = {"Ne": 9, "Xe": None} (or similar) to fix a charge state."""


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
    """Fitted radial profile (out*_averaged.csv): per-fibre mean and std of the physical parameters (PHYS_NAMES)."""
    if not path or not Path(path).exists(): return None
    d = pd.read_csv(path)
    if "radius_um" in d: d = d.sort_values("radius_um")
    M = np.stack([d[f"{k}_mean"].to_numpy(float) for k in PHYS_NAMES], 1)
    S = np.stack([d[f"{k}_std"].to_numpy(float) for k in PHYS_NAMES], 1)
    ok = np.all(np.isfinite(M[:, 1:]), 1)                      # density column is not used any more
    return {"mean": np.nan_to_num(M[ok]).tolist(), "std": np.nan_to_num(S[ok]).tolist(), "source": str(path)}


def make_cfg():
    """Collect all settings into one dict. It is saved to meta.json, so `analyze` needs nothing else."""
    prof = load_profile(PROFILE_CSV)
    if prof is None: print(f"WARNING: profile CSV '{PROFILE_CSV}' not found -> Te/Ti/speeds drawn over the full fit bounds only")
    grid = {"wl_min_nm": float(WL_MIN_NM), "wl_max_nm": float(WL_MAX_NM), "n_pix": int(N_PIX)}
    center = [float(INSTR_CENTER[0]), float(INSTR_CENTER[1])]
    if GRID_FROM_CSV:
        wl, _, s, g = load_ts_csv(GRID_FROM_CSV)
        grid = {"wl_min_nm": float(wl[0]), "wl_max_nm": float(wl[-1]), "n_pix": int(len(wl))}
        if s and g: center = [s, g]
    z = {"scale_jitter": float(Z_SCALE_JITTER)}
    for el in ("Ne", "Xe"):
        if Z_OVERRIDE.get(el) is not None: z[el] = {"override": int(Z_OVERRIDE[el])}
        else:
            f = Path(Z_TABLES[el])
            if not f.exists(): raise SystemExit(Z_TABLE_HELP.format(path=f))
            z[el] = {"file": str(f.resolve()), "md5": hashlib.md5(f.read_bytes()).hexdigest()}
    nb = {k: list(v) for k, v in FIT_BOUNDS.items()}; nb["f_Xe_H"] = [float(XE_H_RATIO_BOUNDS[0]), float(XE_H_RATIO_BOUNDS[1])]; nb["v_rel_e_H"] = list(V_REL_BOUNDS)
    if min(XE_H_RATIO_BOUNDS) < -1 or max(XE_H_RATIO_BOUNDS) > 1: raise SystemExit("XE_H_RATIO_BOUNDS must lie within [-1, 1]")
    dens = {"core": [DENSITY_CORE_CM3[0] * 1e6, DENSITY_CORE_CM3[1] * 1e6], "tail": [DENSITY_TAIL_CM3[0] * 1e6, DENSITY_TAIL_CM3[1] * 1e6],
            "core_frac": float(DENSITY_CORE_FRACTION)}
    shard = max(BATCH, SHARD_SIZE // BATCH * BATCH)
    data = dict(version=2, grid=grid, probe_nm=PROBE_NM, scatter_deg=SCATTER_DEG, z=z, xe_to_h=XE_TO_H,
                fit_bounds={k: list(v) for k, v in FIT_BOUNDS.items()}, net_bounds=nb, density=dens,
                rel_electron=bool(REL_ELECTRON), log_params=list(LOG_PARAMS), sample_log=list(SAMPLE_LOG), log_floor=LOG_FLOOR,
                mix=list(MIX), jitter_floor=JITTER_FLOOR, env_pad_dec=ENV_PAD_DEC, env_pad_frac=ENV_PAD_FRAC,
                instr_center=center, instr_spread=INSTR_SPREAD, profile=prof, shard_size=shard)
    tag = hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return dict(data, tag=tag, seed=SEED, noise_max_frac=NOISE_MAX_FRAC, arch=ARCH, init_values=INIT_VALUES,
                r2_target=R2_TARGET, narrow_max_nfev=NARROW_MAX_NFEV, min_width=MIN_WIDTH, refine_tries=REFINE_TRIES,
                n_spectra=N_SPECTRA, n_val=N_VAL, n_test=N_TEST, batch=BATCH, lr=LR, epochs=EPOCHS, patience=PATIENCE)


# ------------------------------------------------------------------------------------------------ Z lookup table
class ZTable:
    """FLYCHK mean-charge table Zbar(Te [eV], ne [cm^-3]); linear interpolation in (log10 Te, log10 ne), clamped at the edges."""

    def __init__(self, path):
        A = np.loadtxt(path); ne, Te, Z = A[0, 1:], A[1:, 0], A[1:, 1:]
        io, it = np.argsort(ne), np.argsort(Te)
        self.ne, self.Te, self.Z = ne[io], Te[it], Z[it][:, io]
        if self.ne.min() <= 0 or self.Te.min() <= 0: raise ValueError(f"{path}: Te and ne must be > 0")
        if np.any(np.diff(self.ne) <= 0) or np.any(np.diff(self.Te) <= 0): raise ValueError(f"{path}: duplicate Te or ne values")
        self.interp = RegularGridInterpolator((np.log10(self.Te), np.log10(self.ne)), self.Z, method="linear")

    def __call__(self, Te, ne_cm3):
        x = np.log10(np.clip(Te, self.Te[0], self.Te[-1])); y = np.log10(np.clip(ne_cm3, self.ne[0], self.ne[-1]))
        return self.interp(np.stack([x, y], -1))


# ------------------------------------------------------------------------------------------------ physics / scaling
def _voigt(w, sigma, gamma):
    from scipy.special import voigt_profile        # local import so the function pickles cleanly for joblib workers
    return voigt_profile(w, sigma, gamma)


class Ctx:
    """Everything that depends on the configuration: grid, Z lookup, forward model, parameter scaling."""

    def __init__(self, cfg):
        self.cfg = cfg; g = cfg["grid"]
        self.wl_nm = np.linspace(g["wl_min_nm"], g["wl_max_nm"], int(g["n_pix"]))
        self.n_wl, self.wl_m, self.probe_m = len(self.wl_nm), self.wl_nm * 1e-9, cfg["probe_nm"] * 1e-9
        a = np.deg2rad(cfg["scatter_deg"])
        self.pv, self.sv = np.array([1., 0., 0.]), np.array([np.cos(a), np.sin(a), 0.])
        self.khat = (self.sv - self.pv) / np.linalg.norm(self.sv - self.pv)
        self.rel = bool(cfg["rel_electron"]); self.names = net_names(self.rel); self.ename = self.names[7]
        self.fb = {k: tuple(v) for k, v in cfg["fit_bounds"].items()}
        self.nb = {k: tuple(v) for k, v in cfg["net_bounds"].items()}
        self.log = set(cfg["log_params"]); self.n_core = tuple(cfg["density"]["core"])
        self.zmax, self.zov, self.ztab = {"Ne": 10, "Xe": 54}, {}, {}
        for el in ("Ne", "Xe"):
            zc = cfg["z"][el]; self.zov[el] = zc.get("override")
            if self.zov[el] is None: self.ztab[el] = ZTable(zc["file"])
        mass = lambda s: Particle(s).mass.to(u.kg).value
        self.mass = {"Ne": {z: mass(f"Ne-20 {z}+") for z in range(1, 11)}, "Xe": {z: mass(f"Xe-132 {z}+") for z in range(1, 55)}}
        self.h_m, self.xe_to_h = mass("H+"), cfg["xe_to_h"]
        span = self.wl_m.max() - self.wl_m.min()
        self.w_grid = np.linspace(-span / 2, span / 2, self.n_wl)      # zero-centred grid, as spectral_density_model builds it

    # ---- charge states from the lookup (vectorised). zscale: optional (N, 2) multiplicative jitter on Zbar
    def z_ints(self, Te, n_m3, zscale=None):
        Te, n = np.atleast_1d(Te).astype(float), np.atleast_1d(n_m3).astype(float); out = np.empty((len(Te), 2), np.int16)
        for c, el in enumerate(("Ne", "Xe")):
            if self.zov[el] is not None: out[:, c] = int(self.zov[el]); continue
            zb = self.ztab[el](Te, n * 1e-6)
            if zscale is not None: zb = zb * zscale[:, c]
            out[:, c] = np.clip(np.rint(zb), 1, self.zmax[el])
        return out

    def table_report(self):
        te_lo, te_hi = self.fb["T_e_0"]; n_lo, n_hi = self.fb["n"][0] * 1e-6, self.fb["n"][1] * 1e-6
        for el, t in self.ztab.items():
            print(f"  Z table {el}: Te {t.Te[0]:g}-{t.Te[-1]:g} eV, ne {t.ne[0]:.1e}-{t.ne[-1]:.1e} cm^-3 ({len(t.Te)} x {len(t.ne)} points)")
            if t.Te[0] > te_lo or t.Te[-1] < te_hi or t.ne[0] > n_lo or t.ne[-1] < n_hi:
                print(f"  WARNING: {el} table does not cover Te {te_lo:g}-{te_hi:g} eV / ne {n_lo:.0e}-{n_hi:.0e} cm^-3 -> clamped at its edges")
        for el, v in self.zov.items():
            if v is not None: print(f"  Z_{el} fixed to {v}+ (Z_OVERRIDE)")

    # ---- net-space (CNN targets) <-> physical parameters
    def to_phys(self, TH):
        TH = np.atleast_2d(np.asarray(TH, float)); P = TH.copy()
        P[:, 6] = TH[:, 5] * TH[:, 6]                                   # v_Xe = f * v_H   (|v_Xe| <= |v_H|)
        P[:, 7] = TH[:, 5] + TH[:, 7] if self.rel else TH[:, 7]         # v_e  = v_H + v_rel
        return P

    # ---- forward model (max-normalised, instrument function applied, T_i_2 = T_i_1, Xe tied to H). th = PHYSICAL params
    def instr_arr(self, sigma, gamma):
        v = _voigt(self.w_grid, sigma, gamma); v = v / v.max(); return v / v.sum()

    def forward(self, th, z_ne, z_xe, sigma, gamma, ifr=None, norm=True):
        n, Te, Ti0, Ti1, v0, v1, v2, ve, f0 = th; r = self.xe_to_h
        if ifr is None: ifr = np.array([f0, (1 - f0) / (1 + r), (1 - f0) * r / (1 + r)])
        Ti = np.array([Ti0, Ti1, Ti1]) * EV_TO_K
        z = np.array([int(z_ne), 1., int(z_xe)], float)
        m = np.array([self.mass["Ne"][int(z_ne)], self.h_m, self.mass["Xe"][int(z_xe)]])
        iv = np.array([v0, v1, v2])[:, None] * self.khat[None, :]
        _, S = thomson.spectral_density_lite(self.wl_m, self.probe_m, n, np.array([Te * EV_TO_K]), Ti, np.array([1.0]),
                                             ifr, z, m, (ve * self.khat)[None, :], iv, self.pv, self.sv,
                                             instr_func_arr=self.instr_arr(sigma, gamma))
        return S / S.max() if norm else S

    def forward_from_net(self, th_net, sigma, gamma):
        """CNN-style parameters -> spectrum, with Z from the lookup at the given (Te, n)."""
        ph = self.to_phys(th_net)[0]; z = self.z_ints(ph[1], ph[0])[0]
        return self.forward(ph, z[0], z[1], sigma, gamma)

    def ions(self, zi, zj): return [f"Ne-20 {int(zi)}+", "H+", f"Xe-132 {int(zj)}+"]

    def settings(self, zi, zj, sigma, gamma):
        def instr_func(w):
            v = _voigt(w.to(u.m).value, sigma, gamma); return v / v.max()
        return {"probe_wavelength": self.probe_m, "probe_vec": self.pv, "scatter_vec": self.sv, "ions": self.ions(zi, zj),
                "ion_vdir": np.tile(self.khat, (3, 1)), "electron_vdir": np.tile(self.khat, (1, 1)), "instr_func": instr_func}

    # ---- scaling of the CNN targets to [0, 1]
    def to_unit(self, TH):
        U = np.empty(TH.shape, float)
        for j, k in enumerate(self.names):
            lo, hi = self.nb[k]; v = TH[:, j]
            U[:, j] = (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if k in self.log else (v - lo) / (hi - lo)
        return U

    def from_unit(self, U):
        TH = np.empty(U.shape, float)
        for j, k in enumerate(self.names):
            lo, hi = self.nb[k]; x = np.clip(U[:, j], 0, 1)
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


def net_profile(cx, prof):
    """Profile (physical columns) -> mean/std in CNN-target space (f_Xe_H, v_rel computed from the speeds)."""
    M, S = np.array(prof["mean"], float), np.array(prof["std"], float); ix = {k: i for i, k in enumerate(PHYS_NAMES)}
    Mn, Sn = np.zeros((len(M), NP)), np.zeros((len(M), NP))
    for j, k in enumerate(cx.names):
        if k == "n": Mn[:, j] = np.sqrt(cx.n_core[0] * cx.n_core[1])
        elif k == "f_Xe_H":                                     # ratio undefined when v_H ~ 0 -> use 1; spread comes from JITTER_FLOOR
            v1, v2 = M[:, ix["ion_speed_1"]], M[:, ix["ion_speed_2"]]; Mn[:, j] = np.where(np.abs(v1) > 1e3, v2 / np.where(v1 == 0, 1, v1), 1.0)
        elif k == "v_rel_e_H":
            Mn[:, j] = M[:, ix["electron_speed_0"]] - M[:, ix["ion_speed_1"]]; Sn[:, j] = np.hypot(S[:, ix["electron_speed_0"]], S[:, ix["ion_speed_1"]])
        else: Mn[:, j], Sn[:, j] = M[:, ix[k]], S[:, ix[k]]
    return Mn, Sn


# ------------------------------------------------------------------------------------------------ parameter sampler
class Sampler:
    """Draws CNN-space parameters (|f_Xe_H| <= 1, i.e. |v_Xe| <= |v_H|, by construction), density from its own prior, Z jitter and instrument widths."""

    def __init__(self, cx):
        self.cx, cfg = cx, cx.cfg; names = cx.names
        self.islog = np.array([k in set(cfg["sample_log"]) for k in names])
        lo = np.array([cx.nb[k][0] for k in names], float); hi = np.array([cx.nb[k][1] for k in names], float)
        lo = np.where(self.islog & (lo <= 0), cfg["log_floor"], lo)
        self.lo_p, self.hi_p = lo, hi
        self.lo_w, self.hi_w = self._w(lo), self._w(hi)
        self.mix = np.array(cfg["mix"], float)
        self.j_n, self.j_e = names.index("n"), names.index(cx.ename)
        self.j_f = names.index("f_Xe_H")
        prof = cfg.get("profile"); self.PW = None
        if prof:
            Mn, Sn = net_profile(cx, prof); M = np.clip(Mn, lo, hi); self.PW = self._w(M)
            rel = np.where(M > 0, Sn / np.where(M > 0, M, 1), 0)
            est = np.where(self.islog, np.median(rel, 0) / np.log(10), np.median(Sn, 0))
            self.jit = np.maximum(np.array([cfg["jitter_floor"][k] for k in names], float), est)
            rngw = self.PW.max(0) - self.PW.min(0)
            pad = np.where(self.islog, cfg["env_pad_dec"], cfg["env_pad_frac"] * rngw)
            self.env_lo = np.maximum(self.PW.min(0) - pad, self.lo_w); self.env_hi = np.minimum(self.PW.max(0) + pad, self.hi_w)
        else:
            self.mix = np.array([0., 0., 1.])

    def _w(self, P):                                   # physical -> work space (log10 for SAMPLE_LOG params)
        W = np.array(P, float, copy=True); W[..., self.islog] = np.log10(W[..., self.islog]); return W

    def _reflect(self, W):
        W = np.where(W > self.hi_w, 2 * self.hi_w - W, W); W = np.where(W < self.lo_w, 2 * self.lo_w - W, W)
        return np.clip(W, self.lo_w, self.hi_w)

    def sample_density(self, n, rg):
        d = self.cx.cfg["density"]; core = rg.random(n) < d["core_frac"]
        lo = np.where(core, np.log10(d["core"][0]), np.log10(d["tail"][0])); hi = np.where(core, np.log10(d["core"][1]), np.log10(d["tail"][1]))
        return rg.uniform(lo, hi)

    def sample(self, n, rg):
        mode = rg.choice(3, size=n, p=self.mix / self.mix.sum()).astype(np.int8)
        W = rg.uniform(self.lo_w, self.hi_w, (n, NP))                              # mode 2: full bounds
        iA, iB = np.where(mode == 0)[0], np.where(mode == 1)[0]
        if len(iA):                                                                # mode 0: profile + jitter
            nf = len(self.PW)
            if nf == 1: C = np.repeat(self.PW, len(iA), 0)
            else:
                r = rg.uniform(0, nf - 1, len(iA)); i0 = np.minimum(r.astype(int), nf - 2); t = (r - i0)[:, None]
                C = self.PW[i0] * (1 - t) + self.PW[i0 + 1] * t
            W[iA] = self._reflect(C + rg.standard_normal(C.shape) * self.jit)
        if len(iB): W[iB] = rg.uniform(self.env_lo, self.env_hi, (len(iB), NP))     # mode 1: widened envelope
        W[:, self.j_n] = self.sample_density(n, rg)                                 # density: own prior, whatever the mode
        W[:, self.j_e] = rg.uniform(self.lo_w[self.j_e], self.hi_w[self.j_e], n)    # electron (relative) speed: full range
        TH = W.copy(); TH[:, self.islog] = 10 ** W[:, self.islog]
        return TH, mode

    def sample_zscale(self, n, rg):
        j = self.cx.cfg["z"]["scale_jitter"]
        return np.clip(1 + j * rg.standard_normal((n, 2)), 0.2, None) if j > 0 else np.ones((n, 2))

    def sample_instr(self, n, rg):
        a = np.log10(self.cx.cfg["instr_spread"]); cen = np.log10(self.cx.cfg["instr_center"])
        return 10 ** (cen + rg.uniform(-a, a, (n, 2)))


# ------------------------------------------------------------------------------------------------ data generation
def _gen_chunk(cfg, PH, ZI, INS):
    cx = get_ctx(cfg); out = np.zeros((len(PH), cx.n_wl), np.float32); ok = np.zeros(len(PH), bool)
    for i in range(len(PH)):
        try: s = cx.forward(PH[i], ZI[i, 0], ZI[i, 1], INS[i, 0], INS[i, 1])
        except Exception: continue
        if np.all(np.isfinite(s)): out[i], ok[i] = s, True
    return out, ok


def make_shard(cx, sp, n, seed, path, chunk=500):
    rg = np.random.default_rng(seed); parts = []; need = n
    while need > 0:
        th, mode = sp.sample(need, rg); zs = sp.sample_zscale(need, rg); ins = sp.sample_instr(need, rg)
        ph = cx.to_phys(th); zi = cx.z_ints(ph[:, 1], ph[:, 0], zs)                # Z from lookup (+ jitter) at the TRUE Te, n
        res = Parallel(n_jobs=N_JOBS)(delayed(_gen_chunk)(cx.cfg, ph[i:i + chunk], zi[i:i + chunk], ins[i:i + chunk])
                                      for i in range(0, need, chunk))
        x = np.concatenate([r[0] for r in res]); ok = np.concatenate([r[1] for r in res])
        parts.append((x[ok], th[ok], zi[ok], ins[ok], mode[ok])); need -= int(ok.sum())
    cat = lambda j: np.concatenate([p[j] for p in parts])
    tmp = Path(path).with_name(Path(path).stem + ".tmp.npz")
    np.savez(tmp, X=cat(0), TH=cat(1), ZI=cat(2), INS=cat(3), MODE=cat(4)); os.replace(tmp, path)


def data_dir_for(cfg): return Path(OUT_DIR) / f"data_{cfg['tag']}"


def plain_params(cx, init=None):
    """lmfit Parameters in PHYSICAL names (used by selftest to drive plasmapy's own model)."""
    init = {**cx.cfg["init_values"], **(init or {})}; r = cx.xe_to_h; P = Parameters()
    def add(name):
        lo, hi = cx.fb[name]; P.add(name, value=float(np.clip(init[name], lo, hi)), min=lo, max=hi, vary=True)
    add("n"); add("T_i_0"); add("T_i_1"); P.add("T_i_2", expr="T_i_1")
    add("ion_speed_0"); add("ion_speed_1"); add("ion_speed_2"); add("ifract_0")
    P.add("ifract_1", expr=f"(1 - ifract_0) * 1/(1 + {r})"); P.add("ifract_2", expr=f"(1 - ifract_0) * {r}/(1 + {r})")
    add("T_e_0"); add("electron_speed_0")
    return P


def selftest(cx, verbose=True):
    """(1) forward() must reproduce plasmapy's lmfit model; (2) Z lookup sanity; (3) sampler constraints and density prior."""
    rg = np.random.default_rng(1); worst = 0.0
    for t in range(3):
        zi, zj = int(rg.integers(6, 11)), int(rg.integers(10, 27))
        th = np.array([4e24, 60., 400., 20., 3e4, -2e4, -1e4, 5e3, 0.3]); sg, gm = rg.uniform(5e-12, 2e-11, 2)
        P = plain_params(cx, init=dict(zip(PHYS_NAMES, th)))
        mdl = thomson.spectral_density_model(cx.wl_m, cx.settings(zi, zj, sg, gm), P)
        ref = mdl.eval(params=P, wavelengths=cx.wl_m); worst = max(worst, np.abs(cx.forward(th, zi, zj, sg, gm) - ref).max())
    assert worst < 1e-6, f"forward() differs from plasmapy's model by {worst:g}"
    t0 = time.time(); [cx.forward(th, 8, 17, 1e-11, 1e-11) for _ in range(100)]; ms = (time.time() - t0) / 100 * 1e3
    sp = Sampler(cx); TH, _ = sp.sample(5000, rg); PH = cx.to_phys(TH)
    assert np.all(np.abs(PH[:, 6]) <= np.abs(PH[:, 5]) + 1e-6), "constraint |v_Xe| <= |v_H| violated"
    assert PH[:, 6].min() >= cx.fb["ion_speed_2"][0] - 1e-6 and PH[:, 6].max() <= cx.fb["ion_speed_2"][1] + 1e-6, "v_Xe left its bounds"
    ncm = TH[:, 0] * 1e-6; core = np.mean((ncm >= cx.n_core[0] * 1e-6) & (ncm <= cx.n_core[1] * 1e-6))
    if verbose:
        print(f"selftest OK: max |forward - spectral_density_model| = {worst:.2e};  {ms:.2f} ms / spectrum / core")
        print(f"  sampler: |v_Xe| <= |v_H| holds for all {len(TH)} draws; {100 * core:.0f} % of densities inside {cx.n_core[0] * 1e-6:.0e}-{cx.n_core[1] * 1e-6:.0e} cm^-3")
        zz = cx.z_ints(np.array([30., 60., 150.]), np.array([3e24, 3e24, 3e24]))
        print("  Z lookup at ne=3e18 cm^-3, Te=30/60/150 eV -> (Ne, Xe) =", [tuple(int(a) for a in r) for r in zz]); cx.table_report()
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
    """[normal head (2*NN_), beta head (2)] -> [mu(NN_), sigma(NN_), alpha*, beta*]"""
    def __init__(self, sig_eps=2e-3, beta_eps=1e-3, **kw): super().__init__(**kw); self.sig_eps, self.beta_eps = sig_eps, beta_eps
    def call(self, inputs):
        hn, hb = inputs; k = hn.shape[-1] // 2
        return tf.concat([tf.nn.softplus(hn[:, :k]), tf.nn.softplus(hn[:, k:]) + self.sig_eps,
                          tf.nn.softplus(hb) + self.beta_eps], -1)
    def compute_output_shape(self, s): return (s[0][0], s[0][-1] + s[1][-1])
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
    cfg = cx.cfg; a = ARCHS[cfg["arch"]]
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
    return keras.Model({"spectrum": spec, "instr": ins}, DistributionParams(name="dist_params")([hn, hb]))


def make_loss():
    c = 0.5 * np.log(2 * np.pi)
    def ts_loss(y_true, p):
        u_true = tf.cast(y_true, p.dtype)[:, :NP]
        mu, sig, a, b = p[:, :NN_], p[:, NN_:2 * NN_], p[:, 2 * NN_], p[:, 2 * NN_ + 1]
        nll_n = tf.reduce_sum(c + tf.math.log(sig) + 0.5 * tf.square((u_true[:, :NN_] - mu) / sig), -1)
        yb = tf.clip_by_value(u_true[:, NN_], 5e-6, 0.995)
        nll_b = -(tf.math.lgamma(a + b) - tf.math.lgamma(a) - tf.math.lgamma(b) + (a - 1) * tf.math.log(yb) + (b - 1) * tf.math.log1p(-yb))
        mae = tf.reduce_mean(tf.abs(u_true - tf.concat([mu, (a / (a + b))[:, None]], -1)), -1)
        return tf.reduce_mean(nll_n + nll_b + mae)
    return ts_loss


# ------------------------------------------------------------------------------------------------ data feeding
def load_shard(cx, path):
    d = np.load(path)
    Y = cx.to_unit(d["TH"]).astype(np.float32)                                    # 9 scaled targets
    return d["X"], cx.scale_instr(d["INS"][:, 0], d["INS"][:, 1]), Y, d["MODE"], d["TH"], d["INS"], d["ZI"]


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
           tf.TensorSpec((batch, NP), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)


class TimeLimit(keras.callbacks.Callback):
    def __init__(self, minutes): super().__init__(); self.minutes = minutes
    def on_train_begin(self, logs=None): self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        el = (time.time() - self.t0) / 60; print(f"   [{el:.1f} min elapsed]")
        if self.minutes and el > self.minutes: print("   time budget reached -> stopping"); self.model.stop_training = True


class LivePlots(keras.callbacks.Callback):
    """After each epoch redraw PNGs in <OUT_DIR>/live/:  01_loss, 02_metrics (R2 / coverage68), 03_pred_vs_true,
       04_density (true vs predicted n with 68 % bars + interval-width histogram), 05_spectra (simulated spectra + CNN fit)."""
    def __init__(self, cx, X, INS, Y, TH, ZI, out, every=1, Xc=None, n_spec=6):
        super().__init__(); self.cx, self.X, self.INS, self.Y, self.TH, self.ZI, self.every, self.Xc = cx, X, INS, Y, TH, ZI, every, Xc
        self.dir = Path(out) / "live"; (self.dir / "epochs").mkdir(parents=True, exist_ok=True)
        self.spec_idx = np.random.default_rng(123).choice(len(X), min(n_spec, len(X)), replace=False)
        self.h = {k: [] for k in ("epoch", "loss", "val_loss", "lr", "r2", "cov", "nw")}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.every: return
        try: self._update(epoch, logs)
        except Exception as e: print(f"   (live plots skipped: {e})")

    def _update(self, epoch, logs):
        cx, h, U, names = self.cx, self.h, self.Y[:, :NP], self.cx.names
        pr = predict(self.model, cx, self.X, self.INS[:, 0], self.INS[:, 1])
        r2 = [1 - np.sum((U[:, j] - pr["med_u"][:, j]) ** 2) / np.sum((U[:, j] - U[:, j].mean()) ** 2) for j in range(NP)]
        cov = [np.mean((U[:, j] >= pr["lo68_u"][:, j]) & (U[:, j] <= pr["hi68_u"][:, j])) for j in range(NP)]
        wid = np.log10(pr["hi68"][:, 0] / pr["lo68"][:, 0]); lr = logs.get("learning_rate", logs.get("lr", np.nan))
        for k, v in (("epoch", epoch + 1), ("loss", logs.get("loss", np.nan)), ("val_loss", logs.get("val_loss", np.nan)),
                     ("lr", lr), ("r2", r2), ("cov", cov), ("nw", np.median(wid))): h[k].append(v)
        print(f"   [live] mean R2 {np.mean(r2):.3f} | coverage68 {np.mean(cov):.2f} | n: R2 {r2[0]:.2f}, median 68% width {np.median(wid):.2f} dex")
        ep = np.array(h["epoch"])
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
        axs[0].plot(ep, h["loss"], "o-", label="train (noisy batches)"); axs[0].plot(ep, h["val_loss"], "o-", label="validation")
        axs[0].set_yscale("symlog"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss"); axs[0].legend(); axs[0].grid(alpha=.3)
        axs[1].semilogy(ep, h["lr"], "o-"); axs[1].set_xlabel("epoch"); axs[1].set_ylabel("learning rate"); axs[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(self.dir / "01_loss.png", dpi=110); plt.close(fig)
        R, C = np.array(h["r2"]), np.array(h["cov"])
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for j, k in enumerate(names): axs[0].plot(ep, R[:, j], "o-", ms=3, label=k)
        axs[0].set_ylim(-0.1, 1.02); axs[0].set_title("R2 of predicted median (scaled space)"); axs[0].legend(fontsize=7, ncol=2)
        for j, k in enumerate(names): axs[1].plot(ep, C[:, j], "o-", ms=3)
        axs[1].axhline(0.68, color="k", ls="--"); axs[1].set_ylim(0, 1); axs[1].set_title("coverage of the 68 % interval (target 0.68)")
        axs[2].plot(ep, h["nw"], "o-"); axs[2].set_title("density: median 68 % interval width (dex)"); axs[2].set_ylim(bottom=0)
        for a in axs: a.set_xlabel("epoch"); a.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(self.dir / "02_metrics.png", dpi=110); plt.close(fig)
        fig, axs = plt.subplots(3, 3, figsize=(11, 10))
        for ax, (j, k) in zip(axs.ravel(), enumerate(names)):
            ax.scatter(self.TH[:, j], pr["med"][:, j], s=3, alpha=.3, rasterized=True)
            lim = [self.TH[:, j].min(), self.TH[:, j].max()]; ax.plot(lim, lim, "k--", lw=.8); ax.set_title(f"{k}  (R2={r2[j]:.2f})", fontsize=9)
            if k in cx.log: ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("true"); ax.set_ylabel("predicted median")
        fig.suptitle(f"validation subset after epoch {epoch + 1}"); plt.tight_layout(); plt.savefig(self.dir / "03_pred_vs_true.png", dpi=100); plt.close(fig)
        plot_density(cx, self.TH[:, 0], pr, self.dir / "04_density.png", f"after epoch {epoch + 1}")
        n = len(self.spec_idx); nc = min(3, n); nr = int(np.ceil(n / nc)); fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 3.2 * nr), squeeze=False)
        cols = {"wavelength_nm": cx.wl_nm}
        for ax, i in zip(axs.ravel(), self.spec_idx):
            try: fit = cx.forward_from_net(pr["med"][i], self.INS[i, 0], self.INS[i, 1])
            except Exception: fit = np.full(cx.n_wl, np.nan)
            ph = cx.to_phys(pr["med"][i])[0]; zp = cx.z_ints(ph[1], ph[0])[0]; zt = self.ZI[i]
            r2s = 1 - np.var(self.X[i] - fit) / np.var(self.X[i]); clean = self.Xc[i] if self.Xc is not None else self.X[i]
            ax.plot(cx.wl_nm, self.X[i], color="0.75", lw=.7, label="noisy input"); ax.plot(cx.wl_nm, clean, "k", lw=1, label="true (simulated)")
            ax.plot(cx.wl_nm, fit, "r", lw=1, label="CNN fit")
            ax.set_title(f"#{i}  R2={r2s:.3f}  Z true/lookup: Ne {zt[0]}/{zp[0]}  Xe {zt[1]}/{zp[1]}", fontsize=8)
            ax.set_xlabel("nm"); cols[f"data_{i}"], cols[f"true_{i}"], cols[f"fit_{i}"] = self.X[i], clean, fit
        for ax in axs.ravel()[n:]: ax.axis("off")
        axs.ravel()[0].legend(fontsize=7); fig.suptitle(f"simulated validation spectra after epoch {epoch + 1}"); plt.tight_layout()
        plt.savefig(self.dir / "05_spectra.png", dpi=110); plt.savefig(self.dir / "epochs" / f"spectra_epoch_{epoch + 1:03d}.png", dpi=90); plt.close(fig)
        pd.DataFrame(cols).to_csv(self.dir / "epochs" / f"spectra_epoch_{epoch + 1:03d}.csv", index=False)


def plot_density(cx, n_true, pr, path, title=""):
    """Left: true vs predicted n (68 % bars, shaded = focus range).  Right: predicted 68 % width vs the prior's own width."""
    n_p, lo, hi = pr["med"][:, 0], pr["lo68"][:, 0], pr["hi68"][:, 0]; c0, c1 = cx.n_core; sub = np.arange(min(400, len(n_true)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].errorbar(n_true[sub], n_p[sub], yerr=[np.maximum(n_p[sub] - lo[sub], 0), np.maximum(hi[sub] - n_p[sub], 0)], fmt="o", ms=2, alpha=.4, lw=.5)
    lim = [n_true.min(), n_true.max()]; axs[0].plot(lim, lim, "k--", lw=.8); axs[0].axvspan(c0, c1, color="orange", alpha=.15, label="focus range")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("true n (m^-3)"); axs[0].set_ylabel("predicted n, 68 % interval"); axs[0].legend()
    w = np.log10(hi / lo); pw = np.diff(np.percentile(np.log10(n_true), [15.8, 84.2]))[0]
    axs[1].hist(w, bins=40); axs[1].axvline(pw, color="k", ls="--", label=f"prior 68 % width ({pw:.2f} dex)")
    axs[1].set_xlabel("predicted 68 % interval width of n (dex)"); axs[1].set_ylabel("count"); axs[1].legend()
    fig.suptitle(f"density {title}"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


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
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"], clipnorm=1.0), loss=make_loss())
    Xv, Iv, Yv, _, THv, INSv, ZIv = load_shard(cx, dd / "val.npz")
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
        m = min(LIVE_N, nv); cbs.append(LivePlots(cx, Xv[:m], INSv[:m], Yv[:m], THv[:m], ZIv[:m], out, LIVE_EVERY, Xc[:m], LIVE_SPECTRA))
        print(f"live progress figures -> {out / 'live'}/")
    hist = model.fit(train_dataset(cx, files, cfg["batch"], cfg["seed"]), steps_per_epoch=spe, epochs=cfg["epochs"],
                     validation_data=({"spectrum": Xv, "instr": Iv[:nv]}, Yv[:nv]), validation_batch_size=1024,
                     callbacks=cbs, verbose=2)
    plt.figure(figsize=(5, 3)); plt.plot(hist.history["loss"], label="train (noisy)"); plt.plot(hist.history["val_loss"], label="validation")
    plt.yscale("symlog"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "train_curves.png", dpi=150); plt.close()
    print(f"best model saved -> {best}")


# ------------------------------------------------------------------------------------------------ predictions
def unit_quantiles(p, q):
    mu, sig, a, b = p[:, :NN_], p[:, NN_:2 * NN_], p[:, 2 * NN_], p[:, 2 * NN_ + 1]; U = np.empty((len(p), NP))
    U[:, :NN_] = stats.norm.ppf(q, mu, sig); U[:, NN_] = stats.beta.ppf(q, a, b); return np.clip(U, 0, 1)


def predict(model, cx, X, sigma, gamma, bs=1024):
    """Spectra (max-normalised, on the training grid) -> medians and intervals of the CNN targets, in physical units."""
    p = model.predict({"spectrum": np.asarray(X, np.float32), "instr": cx.scale_instr(sigma, gamma)}, batch_size=bs, verbose=0)
    out = dict(raw=p)
    for name, q in (("med", .5), ("lo68", .158), ("hi68", .842), ("lo", .005), ("hi", .995)):
        out[name + "_u"] = unit_quantiles(p, q); out[name] = cx.from_unit(out[name + "_u"])
    return out


def draw_net(cx, praw, k, rg):
    """k draws from ONE spectrum's predicted distribution (heads treated as independent), in physical units of the CNN targets."""
    mu, sig, a, b = praw[:NN_], praw[NN_:2 * NN_], praw[2 * NN_], praw[2 * NN_ + 1]; U = np.empty((k, NP))
    U[:, :NN_] = rg.normal(mu, sig, (k, NN_)); U[:, NN_] = rg.beta(a, b, k)
    return cx.from_unit(np.clip(U, 0, 1))


def cmd_evaluate(cfg, model=None):
    cx = get_ctx(cfg); out = Path(OUT_DIR); dd = data_dir_for(cfg); names = cx.names
    model = model or load_model(out / "ts_model.keras")
    X, I, Y, M, TH, INS, ZI = load_shard(cx, dd / "test.npz")
    Xn = add_noise(X, np.random.default_rng(cfg["seed"] + 7), cfg["noise_max_frac"])
    pr = predict(model, cx, Xn, INS[:, 0], INS[:, 1]); U = Y[:, :NP]; rows = []
    for j, k in enumerate(names):
        r2 = 1 - np.sum((U[:, j] - pr["med_u"][:, j]) ** 2) / np.sum((U[:, j] - U[:, j].mean()) ** 2)
        row = dict(param=k, R2=r2, MAE_pct_of_range=100 * np.mean(np.abs(U[:, j] - pr["med_u"][:, j])),
                   coverage68=np.mean((U[:, j] >= pr["lo68_u"][:, j]) & (U[:, j] <= pr["hi68_u"][:, j])))
        for m, nm in enumerate(MODE_NAMES):
            if (M == m).any(): row[f"MAE%_{nm}"] = 100 * np.mean(np.abs(U[M == m, j] - pr["med_u"][M == m, j]))
        rows.append(row)
    met = pd.DataFrame(rows).set_index("param"); met.round(3).to_csv(out / "eval_metrics.csv")
    print("\nHeld-out test set (noisy). R2/MAE in the CNN's scaled space (log10 for n and temperatures);\n"
          "coverage68 should be ~0.68 for a calibrated network:\n", met.round(3).to_string())
    PHt, PHp = cx.to_phys(TH), cx.to_phys(pr["med"])
    print("\nPhysical velocities backed out from the predicted quantities (compare with the old network's absolute-speed R2):")
    for k in ("ion_speed_2", "electron_speed_0"):
        j = PHYS_NAMES.index(k); e = PHp[:, j] - PHt[:, j]
        print(f"   {k:<17s} R2 = {1 - np.sum(e ** 2) / np.sum((PHt[:, j] - PHt[:, j].mean()) ** 2):.3f}   median |error| = {np.median(np.abs(e)) / 1e3:.1f} km/s")
    c0, c1 = cx.n_core; nt = TH[:, 0]; wid = np.log10(pr["hi68"][:, 0] / pr["lo68"][:, 0]); core = (nt >= c0) & (nt <= c1)
    err = np.abs(np.log10(pr["med"][:, 0]) - np.log10(nt)); pw = np.diff(np.percentile(np.log10(nt), [15.8, 84.2]))[0]
    print(f"\nDensity: median |log10 error| = {np.median(err):.2f} dex overall, {np.median(err[core]):.2f} dex inside {c0 * 1e-6:.0e}-{c1 * 1e-6:.0e} cm^-3"
          f"\n   median predicted 68 % width = {np.median(wid):.2f} dex vs {pw:.2f} dex for the prior itself"
          f"\n   (if these are close, the spectra carry little information on n and the CNN is returning ~the prior)")
    zt = cx.z_ints(PHt[:, 1], PHt[:, 0]).astype(int); zp = cx.z_ints(PHp[:, 1], PHp[:, 0]).astype(int); zres = {}
    for c, el in enumerate(("Ne", "Xe")):
        d = np.abs(zt[:, c] - zp[:, c]); zres[f"Z_{el}"] = ((d == 0).mean(), (d <= 1).mean(), (d <= 2).mean())
        print(f"Z_{el} from lookup at CNN (Te, n) vs lookup at true (Te, n): exact {zres[f'Z_{el}'][0]:.3f} | within +-1 {zres[f'Z_{el}'][1]:.3f} | within +-2 {zres[f'Z_{el}'][2]:.3f}")
    pd.DataFrame(zres, index=["exact", "within_pm1", "within_pm2"]).T.round(4).to_csv(out / "eval_charge_state.csv")
    sub = np.random.default_rng(0).choice(len(U), min(5000, len(U)), replace=False)
    fig, axs = plt.subplots(3, 3, figsize=(11, 10))
    for ax, (j, k) in zip(axs.ravel(), enumerate(names)):
        ax.scatter(TH[sub, j], pr["med"][sub, j], s=3, alpha=.25, rasterized=True)
        lim = [TH[sub, j].min(), TH[sub, j].max()]; ax.plot(lim, lim, "k--", lw=.8); ax.set_title(k)
        if k in cx.log: ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("true"); ax.set_ylabel("predicted median")
    plt.tight_layout(); plt.savefig(out / "eval_scatter.png", dpi=150); plt.close()
    plot_density(cx, nt[sub], {k: v[sub] for k, v in pr.items() if k in ("med", "lo68", "hi68")}, out / "eval_density.png", "(test set)")
    print(f"plots/tables saved in {out}/")


# ------------------------------------------------------------------------------------------------ lmfit refinement
def refine_params(cx, init=None, bounds=None):
    """Free: log10(n), Te, Ti0, Ti1, v_Ne, v_H, f_Xe_H, (v_rel | v_e), ifract_0.
       Derived (real constraints): n = 10**logn,  v_Xe = f_Xe_H * v_H  (|f| <= 1),  v_e = v_H + v_rel."""
    init = {**cx.cfg["init_values"], **(init or {})}; bnd = {**cx.nb, **(bounds or {})}; P = Parameters()
    def add(name):
        lo, hi = bnd[name]; P.add(name, value=float(np.clip(init[name], lo, hi)), min=lo, max=hi, vary=True)
    lo, hi = bnd["n"]; P.add("logn", value=float(np.log10(np.clip(init["n"], lo, hi))), min=float(np.log10(lo)), max=float(np.log10(hi)))
    P.add("n", expr="10**logn")
    for k in ("T_e_0", "T_i_0", "T_i_1", "ion_speed_0", "ion_speed_1", "ifract_0"): add(k)
    add("f_Xe_H"); P.add("ion_speed_2", expr="ion_speed_1 * f_Xe_H")
    add(cx.ename)
    if cx.rel: P.add("electron_speed_0", expr="ion_speed_1 + v_rel_e_H")
    return P


def narrowed_bounds(cx, med, lo, hi):
    """Bounds narrowed to the CNN's 0.5-99.5 % interval (at least MIN_WIDTH of the full range wide)."""
    bnd = {}; mw = cx.cfg["min_width"]
    for j, k in enumerate(cx.names):
        flo, fhi = cx.nb[k]; a, b, m = lo[j], hi[j], med[j]
        if k in cx.log: w = mw * (np.log10(fhi) - np.log10(flo)); a = 10 ** min(np.log10(a), np.log10(m) - w / 2); b = 10 ** max(np.log10(b), np.log10(m) + w / 2)
        else:           w = mw * (fhi - flo);                       a = min(a, m - w / 2);                       b = max(b, m + w / 2)
        bnd[k] = (max(a, flo), min(b, fhi))
    return bnd


def fit_model(cx, sigma, gamma):
    """lmfit model built on the fast forward model; Z is looked up from (T_e, n) on EVERY call, as in your fitting script."""
    def f(wavelengths, n, T_e_0, T_i_0, T_i_1, ion_speed_0, ion_speed_1, ion_speed_2, electron_speed_0, ifract_0, **_):
        z = cx.z_ints([T_e_0], [n])[0]
        th = np.array([n, T_e_0, T_i_0, T_i_1, ion_speed_0, ion_speed_1, ion_speed_2, electron_speed_0, ifract_0])
        return cx.forward(th, z[0], z[1], sigma, gamma)
    return LmModel(f, independent_vars=["wavelengths"])


def fit_spectrum(cx, model, skw, params, method, max_tries=1, max_nfev=None):
    t0 = time.time(); best = None
    for _ in range(max_tries):
        res = model.fit(skw, params=copy.deepcopy(params), wavelengths=cx.wl_m, method=method, max_nfev=max_nfev)
        r2 = 1 - res.residual.var() / np.var(skw)
        if best is None or r2 > best[1]: best = (res, r2)
        if r2 >= cx.cfg["r2_target"]: break
    return (*best, time.time() - t0)


def refine_one(cfg, name, skw, sigma, gamma, med, lo, hi, outdir):
    """lmfit differential evolution with CNN-narrowed bounds; falls back to the full bounds if R2 stays below target."""
    cx = get_ctx(cfg); tgt = cfg["r2_target"]; os.makedirs(outdir, exist_ok=True); model = fit_model(cx, sigma, gamma)
    init, bnd = dict(zip(cx.names, med)), narrowed_bounds(cx, med, lo, hi); used = "CNN-narrowed bounds"
    res, r2, _ = fit_spectrum(cx, model, skw, refine_params(cx, init, bnd), "differential_evolution", cfg["refine_tries"], cfg["narrow_max_nfev"])
    if r2 < tgt:
        res2, r22, _ = fit_spectrum(cx, model, skw, refine_params(cx), "differential_evolution", 4)
        if r22 > r2: res, r2, used = res2, r22, "full bounds (fallback)"
    p = res.params; th = np.array([p[k].value for k in PHYS_NAMES]); zi, zj = [int(z) for z in cx.z_ints(th[1], th[0])[0]]
    ions = cx.ions(zi, zj); wl = cx.wl_nm
    fig, ax = plt.subplots(); ax.plot(wl, skw, label="Data"); ax.plot(wl, res.best_fit, label="Best fit")
    ax.axvline(cfg["probe_nm"], color="red"); ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Skw"); ax.legend()
    ax.set_title(f"{name}: R2={r2:.3f}, {ions[0]}, {ions[2]}", fontsize=9)
    fig.savefig(f"{outdir}/{name}_fit.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame({"wavelength_nm": wl, "data": skw, "best_fit": res.best_fit}).to_csv(f"{outdir}/{name}_fit_spectrum.csv", index=False)
    r, f0 = cx.xe_to_h, th[8]; w = np.array([f0, (1 - f0) / (1 + r), (1 - f0) * r / (1 + r)]) * np.array([zi, 1, zj], float); w /= w.sum()
    plt.figure(); plt.plot(wl, res.best_fit, "k", lw=2, label="Total fit")
    for i in range(3):
        try: plt.plot(wl, cx.forward(th, zi, zj, sigma, gamma, ifr=np.eye(3)[i]) * w[i], label=ions[i])
        except Exception: pass
    plt.legend(); plt.savefig(f"{outdir}/{name}_contributions.png", dpi=200, bbox_inches="tight"); plt.close()
    rows = [{"parameter": k, "value": v.value, "stderr": v.stderr} for k, v in p.items()]
    rows += [{"parameter": "Z_Ne", "value": zi, "stderr": np.nan}, {"parameter": "Z_Xe", "value": zj, "stderr": np.nan}]
    pd.DataFrame(rows).to_csv(f"{outdir}/{name}_params.csv", index=False)
    out = {"fit_R2": r2, "fit_accepted": bool(r2 >= tgt), "fit_bounds_used": used, "fit_Z_Ne": zi, "fit_Z_Xe": zj, "fit_redchi": res.redchi}
    out.update({f"fit_{k}": v for k, v in zip(PHYS_NAMES, th)}); out["fit_f_Xe_H"] = th[6] / th[5] if th[5] != 0 else np.nan; out["fit_v_rel_e_H"] = th[7] - th[5]
    return out


# ------------------------------------------------------------------------------------------------ analyse your spectra
def cmd_analyze(patterns, out_dir=None, refine=False):
    mp = Path(OUT_DIR) / "meta.json"
    if not mp.exists(): raise SystemExit(f"{mp} not found: train first (or set OUT_DIR to the folder holding the trained model)")
    cfg = json.loads(mp.read_text()); cx = get_ctx(cfg); model = load_model(Path(OUT_DIR) / "ts_model.keras"); names_net = cx.names
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
    pr = predict(model, cx, X, SIG, GAM); rows = []; rg = np.random.default_rng(0)
    PH = cx.to_phys(pr["med"]); ZZ = cx.z_ints(PH[:, 1], PH[:, 0]); derived = ("ion_speed_2", "electron_speed_0")
    for i, nm in enumerate(names):
        zi, zj = int(ZZ[i, 0]), int(ZZ[i, 1]); cnn = cx.forward(PH[i], zi, zj, SIG[i], GAM[i])
        dr = cx.to_phys(draw_net(cx, pr["raw"][i], 500, rg))             # 500 draws -> intervals of the derived v_Xe, v_e
        r = {"file": nm, "sigma_m": SIG[i], "gamma_m": GAM[i], "Z_Ne": zi, "Z_Xe": zj, "cnn_R2": 1 - np.var(X[i] - cnn) / np.var(X[i])}
        for j, k in enumerate(names_net): r[k] = pr["med"][i, j]; r[k + "_lo68"] = pr["lo68"][i, j]; r[k + "_hi68"] = pr["hi68"][i, j]
        for k in derived:
            jj = PHYS_NAMES.index(k); r[k] = PH[i, jj]; r[k + "_lo68"], r[k + "_hi68"] = np.percentile(dr[:, jj], [15.8, 84.2])
        rows.append(r)
        fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(cx.wl_nm, X[i], label="data"); ax.plot(cx.wl_nm, cnn, label="CNN median")
        ax.axvline(cfg["probe_nm"], color="red", lw=.6); ax.set_xlabel("nm"); ax.legend(); ax.set_title(f"{nm}: Ne{zi}+  Xe{zj}+ (lookup)", fontsize=9)
        plt.tight_layout(); plt.savefig(out_dir / f"{nm}_cnn.png", dpi=150); plt.close(fig)
        if len(names) <= 5:
            print(f"\n=== {nm}   Z lookup at CNN (Te, n): Ne {zi}+, Xe {zj}+   CNN R2={r['cnn_R2']:.3f}")
            for k in list(names_net) + list(derived):
                s_, un = UNIT_FMT[k]; tag = "  (derived)" if k in derived else ""
                print(f"   {k:<17s} {r[k] * s_:10.4g}  [{r[k + '_lo68'] * s_:.4g}, {r[k + '_hi68'] * s_:.4g}] {un}{tag}")
    df = pd.DataFrame(rows)
    if refine:
        print(f"\nrefining {len(df)} spectra with lmfit differential evolution (CNN-narrowed bounds, Z per iteration)...")
        jobs = [delayed(refine_one)(cfg, nm, X[i], SIG[i], GAM[i], pr["med"][i], pr["lo"][i], pr["hi"][i], str(out_dir / "fits"))
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
