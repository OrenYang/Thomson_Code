#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian (emcee) Thomson-scattering analysis for the Ne / H / Xe multi-ion model,
with OPTIONAL prior narrowing from a trained CNN (ts_train_big_zlookup).

Pipeline per fiber
------------------
0. (USE_CNN) run the trained CNN on the spectrum and turn its predicted distribution into
   padded uniform priors. Safety nets:
     * intervals are padded (PAD_FRAC) and never narrower than MIN_WIDTH_FRAC of the full prior,
     * a side that touches the CNN's own training range is reopened to the full prior,
     * CNN spectrum R^2 < CNN_MIN_R2  -> CNN ignored for this fiber,
     * DE starting fit fails with the narrowed priors -> automatically retried with the full priors,
     * after sampling, warns if the posterior piles up against a CNN-imposed bound.
1. Differential-evolution starting point (R^2 retry loop).
2. Residual-correlation likelihood down-weighting.
3. emcee (DE + DE-snooker moves), run in chunks until the autocorrelation test passes.
4. Systematics sampled as nuisance parameters with Gaussian priors.
5. Outputs: summary table, correlations, corner plots, trace, posterior-predictive fit,
   per-species contributions, chain (.npz), and the priors actually used (.json).

Priors travel to the pool workers inside `ctx` (no env-var / reload tricks), and TensorFlow is
only imported in the main process, so macOS 'spawn' workers stay light.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import logging
import importlib
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import emcee
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution
from scipy.special import voigt_profile
from scipy.stats import skew as _skew
from plasmapy.diagnostics import thomson
from plasmapy.particles import Particle

try:
    import corner
except ImportError:
    corner = None

# ================================================================
#  USER INPUTS
# ================================================================
shot      = 7237
direction = 'n'
FIBERS    = [13]
RUN_IDS   = [1]     # >1 entry = independent repeat runs (different seeds)

TS_ROOT  = '/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis'
PLOT_DIR = f"{TS_ROOT}/output/{shot}"


def csv_path(fiber):
    return f'{TS_ROOT}/ts/0{shot}_TS_shot{direction}/0{shot}_TS_shot{direction}_fiber{fiber}.csv'


# ---- CNN prior narrowing ----------------------------------------------------
USE_CNN        = True
CNN_MODULE     = "thomson_nn"   # file name without .py (must be importable)
CNN_OUT_DIR    = "ts_big_zlookup"         # folder with meta.json + ts_model.keras
USE_CNN_FOR    = ["log10_n", "T_e_0", "T_i_0", "T_i_1", "ion_speed_0", "ion_speed_1",
                  "ion_speed_2", "electron_speed_0", "ifract_0"]   # drop any you don't trust
Q_LO, Q_HI     = 0.5, 99.5     # percentiles of the CNN distribution used as the raw interval
PAD_FRAC       = 0.5           # widen each side by this fraction of the interval (log space for n, T)
MIN_WIDTH_FRAC = 0.15          # never narrower than this fraction of the full prior width
EDGE_U         = 0.03          # CNN quantile within this of its own training edge -> reopen that side
CNN_MIN_R2     = 0.70          # CNN spectrum R^2 below this -> ignore CNN for this fiber
N_CNN_DRAWS    = 20000
FALLBACK_TO_FULL_PRIORS_IF_DE_FAILS = True

# ---- adaptive prior box: pilot runs re-centre each box on the posterior and widen it if the
#      posterior piles up against a movable edge. The final run uses the converged box.
ADAPT_PRIORS      = True
ADAPT_MAX_ROUNDS  = 4       # max pilot runs before the final run
ADAPT_PILOT_STEPS = 6000    # steps per pilot run (second half is used)
ADAPT_PAD         = 1.0     # new box = posterior [0.5, 99.5] % range extended by PAD x that range per side
ADAPT_EDGE_BAND   = 0.05    # "at the edge" = within this fraction of the box width of a bound
ADAPT_EDGE_FRAC   = 0.03    # a side is "hit" if more than this fraction of the posterior is in that band
ADAPT_LIMITS = {            # outermost bounds the box may move to (default = PRIORS). SET FROM PHYSICS.
    "electron_speed_0": (-1.5e6, 1.5e6),
}
BURN_TAU          = 5       # burn-in = BURN_TAU * max autocorrelation time

OUT_DIR = os.path.join(PLOT_DIR, "bayes_cnn" if USE_CNN else "bayes")

# ---- ionization (FLYCHK tables) ---------------------------------------------
Z_TABLES   = {"Ne": "zforNe.txt", "Xe": "zforXe.txt"}
Z_OVERRIDE = {"Ne": None, "Xe": None}
Z_MAX      = {"Ne": 10, "Xe": 54}
CONTINUOUS_Z = True

# ---- scattering geometry ----------------------------------------------------
PROBE_M     = (526.5 * u.nm).to(u.m).value
SCATTER_DEG = 90.0
NOTCH       = None
XE_H_RATIO  = 0.0065

# ---- plot window (PLOTS ONLY: the fit always uses every pixel) -------------------------
# PLOT_WINDOW_NM: "auto" = from the first to the last pixel where a 9-pixel running mean exceeds
#   AUTO_THRESH x peak, extended by PLOT_PAD_NM on each side;  (lo, hi) in nm = fixed;  None = full range.
PLOT_WINDOW_NM     = "auto"
AUTO_THRESH        = 0.02
PLOT_PAD_NM        = 0.5
PLOT_SHOW_NUISANCE = True    # also list the noise + systematics (nuisance) parameters in the fit plot

# ---- FULL priors (uniform between lo, hi); the CNN may narrow these -----------
PRIORS = {
    "log10_n":          (23.0, 27.0),
    "T_e_0":            (0.5, 500.0),
    "T_i_0":            (10.0, 3000.0),
    "T_i_1":            (1.0, 3000.0),
    "ion_speed_0":      (-5e5, 5e5),
    "ion_speed_1":      (-5e5, 5e5),
    "ion_speed_2":      (-5e5, 5e5),
    "electron_speed_0": (-5e5, 5e5),
    "ifract_0":         (0.0, 1),
    "log10_noise":      (-4.0, 0.0),
}
N_PRIOR_LOG10 = None          # optional Gaussian density prior: (mean log10 n[m^-3], sd in dex)
TI_TE_RATIO_BOUNDS = None     # optional hard bound on T_i/T_e, e.g. (0.1, 20)

# ---- systematics: nuisance parameters with Gaussian priors -------------------
MARGINALIZE_SYSTEMATICS = True
NUISANCE = {
    "d_lambda_nm": (0.0, 0.02,  -0.2, 0.2),
    "disp_scale":  (1.0, 0.015,  0.9, 1.1),
    "sigma_scale": (1.0, 0.20,   0.3, 3.0),
    "gamma_scale": (1.0, 0.20,   0.3, 3.0),
    "z_scale":     (1.0, 0.10,   0.5, 1.5),
}

# ---- starting-point optimiser -----------------------------------------------
R2_MIN            = 0.98
DE_TRIES          = 4
DE_KWS            = dict(popsize=15, maxiter=1000, tol=1e-8, polish=True)
SKIP_IF_POOR_FIT  = True

# ---- likelihood ---------------------------------------------------------------
CORRECT_CORRELATED_NOISE = True

# ---- emcee ---------------------------------------------------------------------
N_WALKERS      = None
INIT_SCALE     = 0.02
CHUNK          = 2000
MAX_STEPS      = 60000
TAU_MULT       = 50
N_WORKERS      = None
N_PRED_DRAWS   = 300
SHOW_PLOTS     = True
SKEW_WARNING_THRESHOLD = 0.5
# ================================================================
#  END OF USER INPUTS
# ================================================================

EV_TO_K = 11604.51812155
logger = logging.getLogger("ts_bayes")

# ----------------------------------------------------------------
# Parameter bookkeeping
# ----------------------------------------------------------------
PHYS_NAMES  = ["log10_n", "T_e_0", "T_i_0", "T_i_1", "ion_speed_0", "ion_speed_1",
               "ion_speed_2", "electron_speed_0", "ifract_0"]
NOISE_NAME  = "log10_noise"
NUIS_NAMES  = list(NUISANCE.keys())
NOMINAL     = {k: v[0] for k, v in NUISANCE.items()}
FIT_NAMES   = PHYS_NAMES + [NOISE_NAME] + (NUIS_NAMES if MARGINALIZE_SYSTEMATICS else [])
NDIM        = len(FIT_NAMES)
IDX         = {n: i for i, n in enumerate(FIT_NAMES)}

LO = HI = GMU = GSD = _GAUSS = None   # set by set_priors()


def build_bounds(priors):
    lo, hi = np.zeros(NDIM), np.zeros(NDIM)
    gmu, gsd = np.full(NDIM, np.nan), np.full(NDIM, np.nan)
    for n in PHYS_NAMES + [NOISE_NAME]:
        lo[IDX[n]], hi[IDX[n]] = priors[n]
    if N_PRIOR_LOG10 is not None:
        i = IDX["log10_n"]
        gmu[i], gsd[i] = N_PRIOR_LOG10
        lo[i] = max(lo[i], N_PRIOR_LOG10[0] - 5 * N_PRIOR_LOG10[1])
        hi[i] = min(hi[i], N_PRIOR_LOG10[0] + 5 * N_PRIOR_LOG10[1])
    if MARGINALIZE_SYSTEMATICS:
        for n in NUIS_NAMES:
            mu, sd, l, h = NUISANCE[n]
            gmu[IDX[n]], gsd[IDX[n]] = mu, sd
            lo[IDX[n]], hi[IDX[n]] = l, h
    return lo, hi, gmu, gsd


def set_priors(priors):
    """(Re)build the module-level prior arrays. Called in the main process and in every pool worker."""
    global LO, HI, GMU, GSD, _GAUSS
    LO, HI, GMU, GSD = build_bounds(priors)
    _GAUSS = ~np.isnan(GSD)


set_priors(PRIORS)

PRETTY = {
    "log10_n": r"$\log_{10}\,n_e$ [m$^{-3}$]", "T_e_0": r"$T_e$ [eV]",
    "T_i_0": r"$T_{i,Ne}$ [eV]", "T_i_1": r"$T_{i,H/Xe}$ [eV]",
    "ion_speed_0": r"$v_{Ne}$ [m/s]", "ion_speed_1": r"$v_{H}$ [m/s]",
    "ion_speed_2": r"$v_{Xe}$ [m/s]", "electron_speed_0": r"$v_e$ [m/s]",
    "ifract_0": r"$f_{Ne}$", "log10_noise": r"$\log_{10}\,\sigma_{noise}$",
    "d_lambda_nm": r"$\Delta\lambda_{probe}$ [nm]", "disp_scale": "disp. scale",
    "sigma_scale": r"$\sigma_{Voigt}$ scale", "gamma_scale": r"$\gamma_{Voigt}$ scale",
    "z_scale": r"$Z$ scale",
}


# ----------------------------------------------------------------
# Z lookup (FLYCHK)
# ----------------------------------------------------------------
class ZTable:
    """FLYCHK mean-charge table Zbar(Te [eV], ne [cm^-3]); linear interp in (log10 Te, log10 ne), clamped."""

    def __init__(self, path):
        A = np.loadtxt(path)
        ne, Te, Z = A[0, 1:], A[1:, 0], A[1:, 1:]
        io, it = np.argsort(ne), np.argsort(Te)
        self.ne, self.Te, self.Z = ne[io], Te[it], Z[it][:, io]
        if self.ne.min() <= 0 or self.Te.min() <= 0:
            raise ValueError(f"{path}: Te and ne must be > 0")
        if np.any(np.diff(self.ne) <= 0) or np.any(np.diff(self.Te) <= 0):
            raise ValueError(f"{path}: duplicate Te or ne values")
        self.interp = RegularGridInterpolator(
            (np.log10(self.Te), np.log10(self.ne)), self.Z, method="linear")

    def __call__(self, Te, ne_cm3):
        x = np.log10(np.clip(Te, self.Te[0], self.Te[-1]))
        y = np.log10(np.clip(ne_cm3, self.ne[0], self.ne[-1]))
        return self.interp(np.stack([x, y], -1))


def _load_z_sources():
    src = {}
    for el in ("Ne", "Xe"):
        if Z_OVERRIDE.get(el) is not None:
            src[el] = int(Z_OVERRIDE[el])
        else:
            if not os.path.exists(Z_TABLES[el]):
                raise SystemExit(f"Z lookup table missing: {Z_TABLES[el]}\n"
                                 f"Provide it, or set Z_OVERRIDE['{el}'] to a fixed charge state.")
            src[el] = ZTable(Z_TABLES[el])
    return src


Z_SRC = _load_z_sources()


def get_Z(Te_eV, n_m3, z_scale=1.0):
    """(Z_Ne, Z_Xe) as floats. Table values are multiplied by z_scale; fixed overrides are not."""
    out = []
    for el in ("Ne", "Xe"):
        s = Z_SRC[el]
        if isinstance(s, int):
            z = float(s)
        else:
            zb = float(np.asarray(s(np.atleast_1d(float(Te_eV)),
                                    np.atleast_1d(float(n_m3)) * 1e-6)).ravel()[0])
            z = float(np.clip(zb * z_scale, 1.0, Z_MAX[el]))
        out.append(z if CONTINUOUS_Z else float(np.rint(z)))
    return tuple(out)


@lru_cache(maxsize=None)
def ion_mass(el, z_int):
    sym = {"Ne": "Ne-20", "Xe": "Xe-132"}[el]
    return Particle(f"{sym} {z_int}+").mass.to(u.kg).value


H_MASS = Particle("H+").mass.to(u.kg).value

# ----------------------------------------------------------------
# Scattering geometry
# ----------------------------------------------------------------
probe_vec   = np.array([1, 0, 0])
scatter_vec = np.array([np.cos(np.deg2rad(SCATTER_DEG)), np.sin(np.deg2rad(SCATTER_DEG)), 0])
k_vec = scatter_vec - probe_vec
k_hat = k_vec / np.linalg.norm(k_vec)
ION_NAMES = ["Ne", "H", "Xe"]


# ----------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------
def instr_array(wl_m, sigma, gamma):
    span = wl_m.max() - wl_m.min()
    w = np.linspace(-span / 2, span / 2, wl_m.size)
    v = voigt_profile(w, sigma, gamma)
    return v / v.sum()


def build_v(p):
    f0 = p["ifract_0"]
    return dict(
        n=10.0 ** p["log10_n"], T_e_0=p["T_e_0"],
        T_i_0=p["T_i_0"], T_i_1=p["T_i_1"], T_i_2=p["T_i_1"],
        ion_speed_0=p["ion_speed_0"], ion_speed_1=p["ion_speed_1"], ion_speed_2=p["ion_speed_2"],
        electron_speed_0=p["electron_speed_0"],
        ifract_0=f0,
        ifract_1=(1 - f0) / (1 + XE_H_RATIO),
        ifract_2=(1 - f0) * XE_H_RATIO / (1 + XE_H_RATIO),
    )


def forward(wl_m, v, z_ne, z_xe, instr, probe_m, ifr=None, normalize=True):
    ifract = (np.array([v["ifract_0"], v["ifract_1"], v["ifract_2"]])
              if ifr is None else np.asarray(ifr, float))
    T_i = np.array([v["T_i_0"], v["T_i_1"], v["T_i_2"]]) * EV_TO_K
    z = np.array([z_ne, 1.0, z_xe], float)
    m = np.array([ion_mass("Ne", int(round(z_ne))), H_MASS, ion_mass("Xe", int(round(z_xe)))])
    ion_vel = np.array([v["ion_speed_0"], v["ion_speed_1"], v["ion_speed_2"]])[:, None] * k_hat[None, :]
    el_vel = (v["electron_speed_0"] * k_hat)[None, :]
    kw = {} if NOTCH is None else {"notch": NOTCH}
    _, S = thomson.spectral_density_lite(
        wl_m, probe_m, v["n"], np.array([v["T_e_0"] * EV_TO_K]), T_i, np.array([1.0]),
        ifract, z, m, el_vel, ion_vel, probe_vec, scatter_vec, instr_func_arr=instr, **kw)
    return S / S.max() if normalize else S


def predict(p, ctx, normalize=True, ifr=None):
    """Model spectrum on the data pixels for parameter dict p (physical + nuisance)."""
    v = build_v(p)
    lam0 = ctx["probe_m"]
    wl = lam0 + p["disp_scale"] * (ctx["wl_m"] - lam0)
    instr = instr_array(wl, ctx["sigma"] * p["sigma_scale"], ctx["gamma"] * p["gamma_scale"])
    z_ne, z_xe = get_Z(v["T_e_0"], v["n"], p["z_scale"])
    return forward(wl, v, z_ne, z_xe, instr, lam0 + p["d_lambda_nm"] * 1e-9,
                   ifr=ifr, normalize=normalize)


def to_params(theta):
    p = dict(NOMINAL)
    p.update(zip(FIT_NAMES, theta))
    return p


# ----------------------------------------------------------------
# Prior / likelihood / posterior
# ----------------------------------------------------------------
_CTX = None


def _init_worker(ctx):
    """Runs in the main process and in every pool worker: stores ctx AND rebuilds the (possibly narrowed) priors."""
    global _CTX
    _CTX = ctx
    set_priors(ctx["priors"])


def ratio_ok(p):
    if TI_TE_RATIO_BOUNDS is None:
        return True
    lo, hi = TI_TE_RATIO_BOUNDS
    return all(lo <= p[k] / p["T_e_0"] <= hi for k in ("T_i_0", "T_i_1"))


def log_prior(theta):
    if np.any(theta < LO) or np.any(theta > HI):
        return -np.inf
    return -0.5 * np.sum(((theta[_GAUSS] - GMU[_GAUSS]) / GSD[_GAUSS]) ** 2)


def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    p = to_params(theta)
    if not ratio_ok(p):
        return -np.inf
    try:
        model = predict(p, _CTX)
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(model)):
        return -np.inf
    sig = 10.0 ** p[NOISE_NAME]
    r = model - _CTX["data"]
    ll = -0.5 * np.sum(r ** 2) / sig ** 2 - r.size * np.log(sig)
    return lp + _CTX["like_weight"] * ll


def noise_correlation_weight(resid):
    """1/tau_int of the residual (effective number of independent points correction)."""
    r = np.asarray(resid) - np.mean(resid)
    N = r.size
    acf = np.correlate(r, r, mode="full")[N - 1:]
    acf = acf / acf[0]
    tau = 1.0
    for k in range(1, N // 4):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    return 1.0 / max(tau, 1.0), tau


# ----------------------------------------------------------------
# CNN -> narrowed priors
# ----------------------------------------------------------------
_CNN = None     # (module, cfg, ctx, model), loaded once in the main process

# Bayes parameter -> CNN physical column; widening is done in log space for these
_CNN_COL  = {"log10_n": "n", "T_e_0": "T_e_0", "T_i_0": "T_i_0", "T_i_1": "T_i_1",
             "ion_speed_0": "ion_speed_0", "ion_speed_1": "ion_speed_1",
             "ion_speed_2": "ion_speed_2", "electron_speed_0": "electron_speed_0",
             "ifract_0": "ifract_0"}
_LOG_TEMP = {"T_e_0", "T_i_0", "T_i_1"}


def load_cnn():
    global _CNN
    if _CNN is None:
        cnn = importlib.import_module(CNN_MODULE)       # imports TensorFlow (main process only)
        cfg = json.load(open(os.path.join(CNN_OUT_DIR, "meta.json")))
        cx = cnn.get_ctx(cfg)
        model = cnn.load_model(os.path.join(CNN_OUT_DIR, "ts_model.keras"))
        _CNN = (cnn, cfg, cx, model)
    return _CNN


def _narrow(name, v, full, lo_open, hi_open):
    """v: CNN draws in Bayes-parameter units (log10 n already taken). Returns (lo, hi) inside `full`."""
    Flo, Fhi = full
    islog = name in _LOG_TEMP
    t = np.log10(np.clip(v, 1e-12, None)) if islog else v
    Flo_t, Fhi_t = (np.log10(Flo), np.log10(Fhi)) if islog else (Flo, Fhi)
    lo, hi = np.percentile(t, [Q_LO, Q_HI])
    med = np.median(t)
    w = hi - lo
    lo, hi = lo - PAD_FRAC * w, hi + PAD_FRAC * w
    minw = MIN_WIDTH_FRAC * (Fhi_t - Flo_t)
    if hi - lo < minw:
        lo, hi = med - minw / 2, med + minw / 2
    lo, hi = max(lo, Flo_t), min(hi, Fhi_t)
    if lo_open:
        lo = Flo_t
    if hi_open:
        hi = Fhi_t
    if islog:
        lo, hi = 10 ** lo, 10 ** hi
    return float(lo), float(hi)


def cnn_priors(df, full, rng):
    """Returns (priors dict, info string). Falls back to `full` whenever the CNN can't be trusted."""
    try:
        cnn, cfg, cx, model = load_cnn()
    except Exception as e:
        return dict(full), f"CNN could not be loaded ({type(e).__name__}: {e}) -> full priors"

    wl = df["wavelength_nm"].to_numpy(float)
    y = df["intensity"].to_numpy(float)
    if wl[0] > cx.wl_nm[0] + 0.01 or wl[-1] < cx.wl_nm[-1] - 0.01:
        return dict(full), "spectrum does not cover the CNN wavelength grid -> full priors"
    o = np.argsort(wl)
    yg = np.interp(cx.wl_nm, wl[o], y[o])
    x = yg / yg.max()
    c = cfg["instr_center"]
    s = float(df["sigma"].iloc[0]) if "sigma" in df and np.isfinite(df["sigma"].iloc[0]) else c[0]
    g = float(df["gamma"].iloc[0]) if "gamma" in df and np.isfinite(df["gamma"].iloc[0]) else c[1]

    pr = cnn.predict(model, cx, x[None, :], [s], [g])
    try:
        fit = cx.forward_from_net(pr["med"][0], s, g)
        r2 = 1 - np.var(x - fit) / np.var(x)
    except Exception:
        r2 = -np.inf
    if r2 < CNN_MIN_R2:
        return dict(full), f"CNN spectrum R2 = {r2:.3f} < {CNN_MIN_R2} -> CNN ignored, full priors"

    draws = cx.to_phys(cnn.draw_net(cx, pr["raw"][0], N_CNN_DRAWS, rng))   # columns = cnn.PHYS_NAMES
    out = dict(full)
    for bname in USE_CNN_FOR:
        cname = _CNN_COL[bname]
        v = draws[:, cnn.PHYS_NAMES.index(cname)]
        if bname == "log10_n":
            v = np.log10(np.clip(v, 1e-300, None))
        lo_open = hi_open = False
        if cname in cx.names:                      # directly predicted -> test the CNN's own edges
            ju = cx.names.index(cname)
            lo_open = pr["lo_u"][0, ju] < EDGE_U
            hi_open = pr["hi_u"][0, ju] > 1 - EDGE_U
        out[bname] = _narrow(bname, v, full[bname], lo_open, hi_open)
    return out, f"CNN spectrum R2 = {r2:.3f} -> narrowed priors"


def log_priors_table(pri, full, tag=""):
    for k in PHYS_NAMES:
        mark = "" if pri[k] == full[k] else "   <- narrowed"
        logger.info(f"   {k:<18s} [{pri[k][0]:11.4g}, {pri[k][1]:11.4g}]   "
                    f"(full [{full[k][0]:.4g}, {full[k][1]:.4g}]){mark}")


def box_limits():
    """Outermost bounds the adaptive box may reach: union of PRIORS and ADAPT_LIMITS."""
    out = {}
    for k in PHYS_NAMES:
        lo, hi = PRIORS[k]
        if k in ADAPT_LIMITS:
            lo, hi = min(lo, ADAPT_LIMITS[k][0]), max(hi, ADAPT_LIMITS[k][1])
        out[k] = (lo, hi)
    return out


def _tf(k):
    if k in _LOG_TEMP:
        return np.log10, (lambda t: 10.0 ** t)
    return (lambda t: t), (lambda t: t)


def adapt_box(flat, pri, limits):
    """New box centred on the posterior. Returns (new priors, edge hits on movable sides, truncations at hard limits)."""
    new, hits, trunc = dict(pri), [], []
    for k in PHYS_NAMES:
        lo, hi = pri[k]
        Llo, Lhi = limits[k]
        x = flat[:, IDX[k]]
        band, tol = ADAPT_EDGE_BAND * (hi - lo), 1e-9 * (Lhi - Llo)
        if np.mean(x < lo + band) > ADAPT_EDGE_FRAC:
            (hits if lo - Llo > tol else trunc).append(f"{k} lower ({lo:.4g})")
        if np.mean(x > hi - band) > ADAPT_EDGE_FRAC:
            (hits if Lhi - hi > tol else trunc).append(f"{k} upper ({hi:.4g})")
        f, finv = _tf(k)
        q_lo, q_hi = np.percentile(f(x), [0.5, 99.5])
        w = q_hi - q_lo
        n_lo, n_hi = q_lo - ADAPT_PAD * w, q_hi + ADAPT_PAD * w
        Tlo, Thi = f(Llo), f(Lhi)
        minw = MIN_WIDTH_FRAC * (Thi - Tlo)
        if n_hi - n_lo < minw:
            c = 0.5 * (q_lo + q_hi)
            n_lo, n_hi = c - minw / 2, c + minw / 2
        lo_new, hi_new = finv(max(n_lo, Tlo)), finv(min(n_hi, Thi))
        if lo_new - Llo <= tol:
            lo_new = Llo
        if Lhi - hi_new <= tol:
            hi_new = Lhi
        new[k] = (float(lo_new), float(hi_new))
    return new, hits, trunc


def adapt_priors(ctx, x0, nw, n_workers, rng, run_id):
    """Pilot runs: re-centre / widen the prior box until the posterior no longer hits a movable edge."""
    limits = box_limits()
    pri = dict(ctx["priors"])
    for rnd in range(1, ADAPT_MAX_ROUNDS + 1):
        ctx["priors"] = pri
        set_priors(pri)
        _init_worker(ctx)
        x = np.clip(x0, LO + 1e-9 * (HI - LO), HI - 1e-9 * (HI - LO))
        p0 = init_walkers(x, nw, rng)
        logger.info(f"  [adapt {rnd}/{ADAPT_MAX_ROUNDS}] pilot run, {ADAPT_PILOT_STEPS} steps")
        sampler, _ = run_sampler(p0, ctx, n_workers, seed=1000 * run_id + rnd, max_steps=ADAPT_PILOT_STEPS)
        flat = sampler.get_chain(discard=sampler.iteration // 2, flat=True)
        x0 = np.median(flat, axis=0)
        pri, hits, trunc = adapt_box(flat, pri, limits)
        for t in trunc:
            logger.warning(f"  [adapt {rnd}] posterior truncated by HARD limit: {t} "
                           f"-> widen PRIORS / ADAPT_LIMITS if physically allowed")
        if not hits:
            logger.info(f"  [adapt {rnd}] no edge hits -> box centred on the posterior")
            break
        logger.info(f"  [adapt {rnd}] edge hits: {', '.join(hits)} -> box widened and re-centred")
    else:
        logger.warning("  adaptive boxes still hitting an edge after ADAPT_MAX_ROUNDS "
                       "-> raise ADAPT_MAX_ROUNDS / ADAPT_PAD / ADAPT_LIMITS")
    return pri, x0


def edge_check(flat, pri, limits):
    """Final-run check: warn if the posterior still piles up against any bound."""
    for k in PHYS_NAMES:
        (lo, hi), (Llo, Lhi) = pri[k], limits[k]
        x = flat[:, IDX[k]]
        band, tol = ADAPT_EDGE_BAND * (hi - lo), 1e-9 * (Lhi - Llo)
        for side, hit, bound, movable in (
                ("LOWER", np.mean(x < lo + band) > ADAPT_EDGE_FRAC, lo, lo - Llo > tol),
                ("UPPER", np.mean(x > hi - band) > ADAPT_EDGE_FRAC, hi, Lhi - hi > tol)):
            if hit:
                what = "adjustable (CNN/adaptive) bound -> raise ADAPT_PAD / ADAPT_MAX_ROUNDS or drop it from USE_CNN_FOR" \
                    if movable else "HARD limit -> widen PRIORS / ADAPT_LIMITS if physically allowed"
                logger.warning(f"  !! {k}: posterior piles up at the {side} bound {bound:.4g}: {what}")


# ----------------------------------------------------------------
# Starting point: global optimiser (nuisance at nominal). Uses the current LO/HI.
# ----------------------------------------------------------------
def de_fit(ctx, seed):
    bounds = [(LO[IDX[n]], HI[IDX[n]]) for n in PHYS_NAMES]

    def obj(x):
        p = dict(NOMINAL)
        p.update(zip(PHYS_NAMES, x))
        if not ratio_ok(p):
            return 1e6
        try:
            m = predict(p, ctx)
        except Exception:
            return 1e6
        if not np.all(np.isfinite(m)):
            return 1e6
        return float(np.sum((m - ctx["data"]) ** 2))

    res = differential_evolution(obj, bounds, seed=seed, **DE_KWS)
    p = dict(NOMINAL)
    p.update(zip(PHYS_NAMES, res.x))
    best = predict(p, ctx)
    r2 = 1 - np.var(best - ctx["data"]) / np.var(ctx["data"])
    return p, best, r2


def de_with_retries(ctx, run_id):
    best_r2, best = -np.inf, None
    for t in range(DE_TRIES):
        p_de, fit_de, r2 = de_fit(ctx, seed=run_id * 100 + t)
        logger.info(f"  DE try {t + 1}/{DE_TRIES}: R^2 = {r2:.4f}")
        if r2 > best_r2:
            best_r2, best = r2, (p_de, fit_de)
        if r2 >= R2_MIN:
            break
    return best[0], best[1], best_r2


# ----------------------------------------------------------------
# emcee driver
# ----------------------------------------------------------------
def init_walkers(x0, nwalkers, rng):
    step = np.where(_GAUSS, 0.5 * np.nan_to_num(GSD),
                    INIT_SCALE * np.maximum(np.abs(x0), 0.02 * (HI - LO)))
    p0, attempts, scale = [], 0, 1.0
    while len(p0) < nwalkers:
        cand = x0 + scale * step * rng.standard_normal(NDIM)
        cand = np.clip(cand, LO + 1e-12 * (HI - LO), HI - 1e-12 * (HI - LO))
        if np.isfinite(log_prob(cand)):
            p0.append(cand)
        attempts += 1
        if attempts % 500 == 0:
            scale *= 0.5
    return np.array(p0)


def run_sampler(p0, ctx, n_workers, seed, max_steps=None):
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    nwalkers = p0.shape[0]
    pool = None
    if n_workers > 1:
        pool = Pool(n_workers, initializer=_init_worker, initargs=(ctx,))
    else:
        _init_worker(ctx)
    try:
        sampler = emcee.EnsembleSampler(nwalkers, NDIM, log_prob, pool=pool, moves=moves)
        sampler.random_state = np.random.RandomState(seed).get_state()
        state, old_tau, converged = p0, np.full(NDIM, np.inf), False
        while sampler.iteration < (max_steps or MAX_STEPS):
            state = sampler.run_mcmc(state, CHUNK, progress=False)
            tau = sampler.get_autocorr_time(tol=0)
            acc = np.mean(sampler.acceptance_fraction)
            logger.info(f"  iter {sampler.iteration:6d}  max tau={np.nanmax(tau):8.1f}  "
                        f"iter/tau={sampler.iteration / np.nanmax(tau):6.1f}  acc={acc:.3f}")
            converged = bool(np.all(tau * TAU_MULT < sampler.iteration)
                             and np.all(np.abs(old_tau - tau) / tau < 0.01))
            if converged:
                break
            old_tau = tau
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return sampler, converged


# ----------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------
def derived_quantities(samples, max_n=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(samples) > max_n:
        samples = samples[rng.choice(len(samples), max_n, replace=False)]
    cols = {k: [] for k in ["Z_Ne", "Z_Xe", "n_e_1e18_cm3", "ion_drift_Ne_minus_e_kms",
                            "Ti0_over_Te", "Ti1_over_Te"]}
    for th in samples:
        p = to_params(th)
        n_m3 = 10.0 ** p["log10_n"]
        z_ne, z_xe = get_Z(p["T_e_0"], n_m3, p["z_scale"])
        cols["Z_Ne"].append(z_ne)
        cols["Z_Xe"].append(z_xe)
        cols["n_e_1e18_cm3"].append(n_m3 * 1e-6 / 1e18)
        cols["ion_drift_Ne_minus_e_kms"].append((p["ion_speed_0"] - p["electron_speed_0"]) / 1e3)
        cols["Ti0_over_Te"].append(p["T_i_0"] / p["T_e_0"])
        cols["Ti1_over_Te"].append(p["T_i_1"] / p["T_e_0"])
    return {k: np.array(v) for k, v in cols.items()}


def summarize(columns, map_vals=None):
    rows = []
    for name, x in columns.items():
        q16, q50, q84 = np.percentile(x, [15.865, 50, 84.135])
        sk = float(_skew(x))
        rows.append(dict(parameter=name, median=q50, minus_68=q50 - q16, plus_68=q84 - q50,
                         mean=float(np.mean(x)), std=float(np.std(x)), skew=sk,
                         skew_flag="SKEWED: use percentiles, not mean+/-std"
                         if abs(sk) > SKEW_WARNING_THRESHOLD else "",
                         MAP=np.nan if map_vals is None or name not in map_vals else map_vals[name]))
    return pd.DataFrame(rows)


def split_half_check(sampler, burn):
    chain = sampler.get_chain(discard=burn)
    h = chain.shape[0] // 2
    a, b = chain[:h].reshape(-1, NDIM), chain[h:].reshape(-1, NDIM)
    return np.abs(np.median(a, 0) - np.median(b, 0)) / (0.5 * (a.std(0) + b.std(0)) + 1e-300)


# ----------------------------------------------------------------
# Plots
# ----------------------------------------------------------------
def _finish(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _zoom(ax, ctx):
    xl = ctx.get("plot_xlim")
    if xl is not None:
        ax.set_xlim(*xl)


def plot_trace(sampler, burn, path):
    names = PHYS_NAMES + [NOISE_NAME]
    chain = sampler.get_chain()
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.5 * len(names)), sharex=True)
    for ax, nm in zip(axes, names):
        ax.plot(chain[:, :16, IDX[nm]], alpha=0.35, lw=0.6)
        ax.axvline(burn, color="k", ls="--", lw=1)
        ax.set_ylabel(nm, fontsize=7)
    axes[-1].set_xlabel("step  (dashed = burn-in cut)")
    _finish(fig, path)


def plot_corners(flat, out_prefix):
    if corner is None:
        logger.warning("`corner` not installed -> skipping corner plots (pip install corner)")
        return
    names = PHYS_NAMES + [NOISE_NAME]
    fig = corner.corner(flat[:, [IDX[n] for n in names]], labels=[PRETTY[n] for n in names],
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g",
                        title_kwargs={"fontsize": 7}, label_kwargs={"fontsize": 8})
    _finish(fig, f"{out_prefix}_corner_physical.png")
    if MARGINALIZE_SYSTEMATICS:
        names = ["log10_n", "T_e_0", "T_i_0", "ion_speed_0", "electron_speed_0"] + NUIS_NAMES
        fig = corner.corner(flat[:, [IDX[n] for n in names]], labels=[PRETTY[n] for n in names],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g",
                            title_kwargs={"fontsize": 7}, label_kwargs={"fontsize": 8})
        _finish(fig, f"{out_prefix}_corner_systematics.png")


def plot_fit(ctx, de_best, flat, map_theta, summ, out_prefix, rng, fiber):
    idx = rng.choice(len(flat), min(N_PRED_DRAWS, len(flat)), replace=False)
    curves = np.array([predict(to_params(flat[i]), ctx) for i in idx])
    lo95, lo68, med, hi68, hi95 = np.percentile(curves, [2.5, 15.865, 50, 84.135, 97.5], axis=0)
    wl_nm = ctx["wl_m"] * 1e9

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.subplots_adjust(right=0.70)
    ax.fill_between(wl_nm, lo95, hi95, color="orange", alpha=0.20, label="95 % posterior band")
    ax.fill_between(wl_nm, lo68, hi68, color="orange", alpha=0.40, label="68 % posterior band")
    ax.plot(wl_nm, ctx["data"], "k.", ms=3, label="Data")
    ax.plot(wl_nm, med, "r-", lw=1.6, label="Posterior-median spectrum")
    ax.plot(wl_nm, predict(to_params(map_theta), ctx), "g-", lw=1.0, alpha=0.8, label="MAP spectrum")
    ax.plot(wl_nm, de_best, "b--", lw=1.0, alpha=0.7, label="DE starting fit")
    ax.axvline(ctx["probe_m"] * 1e9, color="red", lw=1)

    def _g(x, nsig):                      # 3 sig. figs, but never scientific notation for values >= 100
        return f"{x:.0f}" if abs(x) >= 100 else f"{x:.{nsig}g}"

    def pm(name, scale=1.0):
        r = summ.loc[summ.parameter == name].iloc[0]
        return (f"{_g(r['median'] * scale, 3)} (+{_g(r['plus_68'] * scale, 2)}"
                f"/-{_g(r['minus_68'] * scale, 2)})")
    zn = summ.loc[summ.parameter == "Z_Ne", "median"].iloc[0]
    zx = summ.loc[summ.parameter == "Z_Xe", "median"].iloc[0]
    lines = [
        r"$n_e$ = " + pm("n_e_1e18_cm3") + r" $10^{18}$ cm$^{-3}$",
        r"$T_e$ = " + pm("T_e_0") + " eV",
        r"$T_{i,Ne}$ = " + pm("T_i_0") + " eV",
        r"$T_{i,H/Xe}$ = " + pm("T_i_1") + " eV",
        r"$v_{Ne}$ = " + pm("ion_speed_0", 1e-3) + " km/s",
        r"$v_{H}$ = " + pm("ion_speed_1", 1e-3) + " km/s",
        r"$v_{Xe}$ = " + pm("ion_speed_2", 1e-3) + " km/s",
        r"$v_e$ = " + pm("electron_speed_0", 1e-3) + " km/s",
        r"$f_{Ne}$ = " + pm("ifract_0"),
        f"$Z_{{Ne}}$ = {zn:.2f},  $Z_{{Xe}}$ = {zx:.2f}",
    ]
    if PLOT_SHOW_NUISANCE:
        lines += ["", r"$\log_{10}\,\sigma_{noise}$ = " + pm(NOISE_NAME)]
        if MARGINALIZE_SYSTEMATICS:
            lines += [r"$\Delta\lambda_{probe}$ = " + pm("d_lambda_nm") + " nm",
                      "disp. scale = " + pm("disp_scale"),
                      r"$\sigma_{Voigt}$ scale = " + pm("sigma_scale"),
                      r"$\gamma_{Voigt}$ scale = " + pm("gamma_scale"),
                      "$Z$ scale = " + pm("z_scale")]
    ax.text(1.02, 1.0, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.9, ec="0.7"))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Skw (normalised)")
    ax.set_title(f"Shot {shot} fiber {fiber}: posterior predictive spectrum")
    _zoom(ax, ctx)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    _finish(fig, f"{out_prefix}_fit.png")
    return wl_nm, med, lo68, hi68


def plot_contributions(ctx, theta_med, out_prefix):
    p = to_params(theta_med)
    v = build_v(p)
    lam0 = ctx["probe_m"]
    wl = lam0 + p["disp_scale"] * (ctx["wl_m"] - lam0)
    instr = instr_array(wl, ctx["sigma"] * p["sigma_scale"], ctx["gamma"] * p["gamma_scale"])
    z_ne, z_xe = get_Z(v["T_e_0"], v["n"], p["z_scale"])
    probe = lam0 + p["d_lambda_nm"] * 1e-9
    total = forward(wl, v, z_ne, z_xe, instr, probe, normalize=False)
    ifr = np.array([v["ifract_0"], v["ifract_1"], v["ifract_2"]])
    w = ifr * np.array([z_ne, 1.0, z_xe])
    w /= w.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ctx["wl_m"] * 1e9, total / total.max(), "k", lw=2, label="Total (median params)")
    for i in range(3):
        s = forward(wl, v, z_ne, z_xe, instr, probe, ifr=np.eye(3)[i], normalize=False)
        ax.plot(ctx["wl_m"] * 1e9, w[i] * s / total.max(),
                label=f"{ION_NAMES[i]} ({[z_ne, 1, z_xe][i]:.2f}+)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Skw (normalised to total max)")
    _zoom(ax, ctx)
    ax.legend()
    _finish(fig, f"{out_prefix}_contributions.png")


def plot_window(wl_nm, data):
    """(lo, hi) x-range in nm for the plots, or None for the full range. Plots only; the fit uses every pixel."""
    if PLOT_WINDOW_NM is None:
        return None
    if isinstance(PLOT_WINDOW_NM, str):                      # "auto"
        sm = np.convolve(data, np.ones(9) / 9, mode="same")
        w = wl_nm[sm > AUTO_THRESH * sm.max()]
        return float(w.min() - PLOT_PAD_NM), float(w.max() + PLOT_PAD_NM)
    return tuple(float(v) for v in PLOT_WINDOW_NM)


# ----------------------------------------------------------------
# One full run for one fiber / one seed
# ----------------------------------------------------------------
def analyse(df, fiber, run_id, narrowed, narrow_info):
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"fiber{fiber}_run{run_id}"
    out_prefix = os.path.join(OUT_DIR, tag)
    fh = logging.FileHandler(f"{out_prefix}.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    try:
        rng = np.random.default_rng(1000 + run_id)
        full = {k: tuple(v) for k, v in PRIORS.items()}
        priors = dict(narrowed)

        wl_m = df["wavelength_nm"].to_numpy() * 1e-9
        data = df["intensity"].to_numpy(float)
        data = data / data.max()
        ctx = dict(wl_m=wl_m, data=data, sigma=float(df["sigma"].iloc[0]),
                   gamma=float(df["gamma"].iloc[0]), probe_m=PROBE_M, like_weight=1.0,
                   priors=priors)
        ctx["plot_xlim"] = plot_window(wl_m * 1e9, data)

        logger.info("=" * 70)
        logger.info(f"Shot {shot} fiber {fiber} run {run_id}   N_data={wl_m.size}   ndim={NDIM}")
        if ctx["plot_xlim"] is not None:
            logger.info(f"Plot x-range (plots only): {ctx['plot_xlim'][0]:.3f}-{ctx['plot_xlim'][1]:.3f} nm")
        logger.info(f"Systematics marginalised: {MARGINALIZE_SYSTEMATICS}   continuous Z: {CONTINUOUS_Z}")
        logger.info(f"Priors: {narrow_info}")
        log_priors_table(priors, full)

        # ---- 1. starting point (with fallback to the full priors) -----------------
        set_priors(priors)
        p_de, fit_de, best_r2 = de_with_retries(ctx, run_id)
        if (best_r2 < R2_MIN and priors != full and FALLBACK_TO_FULL_PRIORS_IF_DE_FAILS):
            logger.warning(f"Best R^2 = {best_r2:.4f} < {R2_MIN} with CNN-narrowed priors "
                           f"-> retrying DE with the FULL priors (CNN narrowing dropped for this fiber)")
            priors = dict(full)
            ctx["priors"] = priors
            set_priors(priors)
            p_de, fit_de, best_r2 = de_with_retries(ctx, run_id)
        if best_r2 < R2_MIN:
            logger.warning(f"Best R^2 = {best_r2:.4f} < {R2_MIN}")
            if SKIP_IF_POOR_FIT:
                logger.warning("SKIP_IF_POOR_FIT=True -> skipping this run")
                return None
        json.dump({k: list(v) for k, v in priors.items()}, open(f"{out_prefix}_priors_used.json", "w"), indent=1)

        # ---- 2. noise model ------------------------------------------------------
        resid = fit_de - data
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if CORRECT_CORRELATED_NOISE:
            ctx["like_weight"], tau_noise = noise_correlation_weight(resid)
            logger.info(f"  residual rms = {rms:.4g};  noise autocorr. length tau = {tau_noise:.2f} pixels "
                        f"-> likelihood weight {ctx['like_weight']:.3f} "
                        f"(error bars widened by ~{np.sqrt(tau_noise):.2f}x)")
        else:
            logger.info(f"  residual rms = {rms:.4g};  correlated-noise correction OFF")

        # ---- 3. MCMC ---------------------------------------------------------------
        x0 = np.array([p_de[n] if n in p_de else (np.log10(rms) if n == NOISE_NAME else NOMINAL[n])
                       for n in FIT_NAMES])
        x0 = np.clip(x0, LO + 1e-9 * (HI - LO), HI - 1e-9 * (HI - LO))
        nw = N_WALKERS or int(4 * NDIM + (4 * NDIM) % 2)
        nw = max(nw + nw % 2, 2 * NDIM + (2 * NDIM) % 2)
        n_workers = N_WORKERS or cpu_count()
        if ADAPT_PRIORS:
            priors, x0 = adapt_priors(ctx, x0, nw, n_workers, rng, run_id)
            ctx["priors"] = priors
            set_priors(priors)
            x0 = np.clip(x0, LO + 1e-9 * (HI - LO), HI - 1e-9 * (HI - LO))
            json.dump({k: list(v) for k, v in priors.items()},
                      open(f"{out_prefix}_priors_used.json", "w"), indent=1)
            logger.info("Final prior box after adaptation:")
            log_priors_table(priors, full)
        _init_worker(ctx)            # main process: ctx + final priors
        p0 = init_walkers(x0, nw, rng)
        logger.info(f"  emcee: {nw} walkers, {n_workers} worker(s), chunks of {CHUNK}, max {MAX_STEPS} steps")
        sampler, converged = run_sampler(p0, ctx, n_workers, seed=run_id)

        tau = sampler.get_autocorr_time(tol=0)
        it = sampler.iteration
        tau_max = np.nanmax(tau)
        burn = int(min(BURN_TAU * tau_max, it // 2)) if np.isfinite(tau_max) else it // 2
        thin = max(1, int(0.5 * np.nanmin(tau))) if np.all(np.isfinite(tau)) else 1
        flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
        lnp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
        n_indep = (it - burn) * nw / tau_max
        logger.info(f"  converged = {converged};  iterations = {it};  burn = {burn};  thin = {thin};  "
                    f"kept samples = {len(flat)};  ~independent samples = {n_indep:.0f}")
        if not converged:
            logger.warning("Chains did NOT satisfy the iteration > %d*tau test. Increase MAX_STEPS, "
                           "or treat these results as provisional." % TAU_MULT)
        z_half = split_half_check(sampler, burn)
        for nm, zz in zip(FIT_NAMES, z_half):
            if zz > 0.2:
                logger.warning(f"  split-half drift: {nm} median moved by {zz:.2f} std between halves")

        edge_check(flat, priors, box_limits())

        # ---- 4. summaries -------------------------------------------------------------
        map_theta = flat[np.argmax(lnp)]
        theta_med = np.median(flat, axis=0)
        derived = derived_quantities(flat, rng=rng)
        columns = {n: flat[:, IDX[n]] for n in FIT_NAMES}
        columns.update(derived)
        map_vals = {n: map_theta[IDX[n]] for n in FIT_NAMES}
        summ = summarize(columns, map_vals)
        summ.insert(1, "tau_autocorr", [tau[IDX[n]] if n in IDX else np.nan for n in summ.parameter])
        summ.to_csv(f"{out_prefix}_summary.csv", index=False)
        logger.info("\n" + summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

        corr = pd.DataFrame(flat, columns=FIT_NAMES).corr()
        corr.to_csv(f"{out_prefix}_correlation.csv")
        strong = [(a, b, corr.loc[a, b]) for i, a in enumerate(FIT_NAMES)
                  for b in FIT_NAMES[i + 1:] if abs(corr.loc[a, b]) > 0.7]
        for a, b, r in sorted(strong, key=lambda t: -abs(t[2])):
            logger.info(f"  strong correlation: {a} vs {b}: rho = {r:+.2f}")
        if MARGINALIZE_SYSTEMATICS:
            for n in PHYS_NAMES:
                for m_ in NUIS_NAMES:
                    if abs(corr.loc[n, m_]) > 0.3:
                        logger.info(f"  systematic sensitivity: {n} <-> {m_}: rho = {corr.loc[n, m_]:+.2f}")

        np.savez_compressed(f"{out_prefix}_chain.npz", samples=flat, log_prob=lnp,
                            names=np.array(FIT_NAMES), tau=tau, converged=converged)

        # ---- 5. plots + spectrum csv ----------------------------------------------------
        plot_trace(sampler, burn, f"{out_prefix}_trace.png")
        plot_corners(flat, out_prefix)
        wl_nm, med, lo68, hi68 = plot_fit(ctx, fit_de, flat, map_theta, summ, out_prefix, rng, fiber)
        plot_contributions(ctx, theta_med, out_prefix)
        pd.DataFrame({"wavelength_nm": wl_nm, "data": data, "de_fit": fit_de,
                      "posterior_median": med, "lo68": lo68, "hi68": hi68}
                     ).to_csv(f"{out_prefix}_fit_spectrum.csv", index=False)
        logger.info(f"Saved everything to {OUT_DIR}  (prefix {tag})")
        return summ
    finally:
        set_priors(PRIORS)           # restore defaults for the next fiber
        logger.removeHandler(fh)
        fh.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)
    full = {k: tuple(v) for k, v in PRIORS.items()}
    rng = np.random.default_rng(0)
    for fiber in FIBERS:
        df = pd.read_csv(csv_path(fiber))
        if USE_CNN:
            narrowed, info = cnn_priors(df, full, rng)
        else:
            narrowed, info = dict(full), "CNN disabled -> full priors"
        logger.info(f"fiber {fiber}: {info}")
        for run_id in RUN_IDS:
            analyse(df, fiber, run_id, narrowed, info)


if __name__ == "__main__":        # required: multiprocessing on macOS uses 'spawn'
    main()
