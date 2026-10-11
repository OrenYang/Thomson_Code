#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian (emcee) Thomson-scattering analysis for the Ne / H / Xe multi-ion model.

What this does, in order
------------------------
1. Loads one fiber's spectrum + the Voigt instrument parameters from the CSV.
2. Finds a starting point with a global optimiser (scipy differential_evolution,
   same idea as your lmfit DE fit, with the R^2 retry loop).
3. Estimates how correlated the residual noise is and down-weights the
   likelihood accordingly (so error bars are not over-confident).
4. Samples the full posterior with emcee (DE + DE-snooker moves, which cope
   much better with strongly correlated parameters than the default stretch
   move), running in chunks until the autocorrelation-time test says the
   chains are long enough.
5. Systematics are sampled *inside* the MCMC as nuisance parameters with priors
   (laser wavelength offset, wavelength-axis scale, Voigt sigma/gamma scale,
   Z-table scale), so they are marginalised rather than bolted on by refitting.
6. Saves: summary table (median, +/- 68 %, skew, MAP), correlation matrix,
   corner plots, trace plot, posterior-predictive spectrum, per-species
   contribution plot, and the thinned chain (.npz).

Everything you normally edit is in the USER INPUTS block.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")        # one thread per worker process,
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # otherwise the pool oversubscribes
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
import datetime
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
except ImportError:           # corner plots are skipped if the package is missing
    corner = None

# ================================================================
#  USER INPUTS
# ================================================================
shot      = 7237
fiber     = 15
direction = 'n'
RUN_IDS   = [1]     # >1 entry = independent repeat runs (different seeds).
                    # Comparing their posteriors is a good convergence check.

CSV_PATH = (f'/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/ts/'
            f'0{shot}_TS_shot{direction}/0{shot}_TS_shot{direction}_fiber{fiber}.csv')
PLOT_DIR = f"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/output/{shot}"
OUT_DIR  = os.path.join(PLOT_DIR, "bayes")

# ---- ionization (FLYCHK tables, same layout as before) ---------------------
Z_TABLES   = {"Ne": "zforNe.txt", "Xe": "zforXe.txt"}
Z_OVERRIDE = {"Ne": None, "Xe": None}     # int -> fixed charge state (not scaled by z_scale)
Z_MAX      = {"Ne": 10, "Xe": 54}
CONTINUOUS_Z = True    # True : Z = Zbar*z_scale kept as a float (smooth likelihood, recommended)
                       # False: Z is rounded to an integer like the old code (likelihood is
                       #        piecewise-constant in Z, which MCMC handles poorly)

# ---- scattering geometry ---------------------------------------------------
PROBE_M     = (526.5 * u.nm).to(u.m).value
SCATTER_DEG = 90.0
NOTCH       = None    # e.g. (np.array([526.25, 526.6]) * u.nm).to(u.m).value
XE_H_RATIO  = 0.0065  # fixed Xe/H ion-fraction ratio (as in your ifract_1 / ifract_2 expressions)

# ---- priors on the physical parameters (uniform between lo, hi) -------------
# density is sampled as log10(n [m^-3]); 23..27 is your old 1e23..1e27 m^-3 box.
PRIORS = {
    "log10_n":          (23.0, 27.0),
    "T_e_0":            (0.5, 500.0),     # eV
    "T_i_0":            (10.0, 3000.0),   # eV  (Ne)
    "T_i_1":            (1.0, 3000.0),     # eV  (H and Xe share this, as before)
    "ion_speed_0":      (-5e5, 5e5),      # m/s along k
    "ion_speed_1":      (-5e5, 5e5),
    "ion_speed_2":      (-5e5, 5e5),
    "electron_speed_0": (-5e5, 5e5),
    "ifract_0":         (0.0, 1),
    "log10_noise":      (-4.0, 0.0),      # log10 of the white-noise sigma (data is max-normalised)
}
# Optional informative density prior, e.g. from interferometry: (mean log10 n[m^-3], sd in dex).
# Example: n_e ~ 5e17 cm^-3 +/- factor 3  ->  N_PRIOR_LOG10 = (np.log10(5e17*1e6), 0.5)
N_PRIOR_LOG10 = None
# Optional hard bound on T_i/T_e (for both ion temperatures), e.g. (0.1, 20). None = off.
TI_TE_RATIO_BOUNDS = None

# ---- systematics: nuisance parameters with Gaussian priors ------------------
#   name:          (mean, sd, hard_lo, hard_hi)
# Same dispersions as your old MC script (laser 0.02 nm absolute, widths 20 %, dispersion 1.5 %, Z 10 %).
# n_e is NOT here any more: it is a fitted parameter.
MARGINALIZE_SYSTEMATICS = True    # False -> nuisance fixed at nominal (statistical-only error bars)
NUISANCE = {
    "d_lambda_nm": (0.0, 0.02,  -0.2, 0.2),    # probe wavelength offset [nm]
    "disp_scale":  (1.0, 0.015,  0.9, 1.1),    # wavelength-axis scale about the probe line
    "sigma_scale": (1.0, 0.20,   0.3, 3.0),    # Voigt sigma multiplier
    "gamma_scale": (1.0, 0.20,   0.3, 3.0),    # Voigt gamma multiplier
    "z_scale":     (1.0, 0.10,   0.5, 1.5),    # multiplier on the table Zbar
}

# ---- starting-point optimiser ----------------------------------------------
R2_MIN            = 0.98
DE_TRIES          = 4
DE_KWS            = dict(popsize=15, maxiter=1000, tol=1e-8, polish=True)
SKIP_IF_POOR_FIT  = True     # old behaviour: skip the fiber if R^2 < R2_MIN after DE_TRIES

# ---- likelihood -------------------------------------------------------------
CORRECT_CORRELATED_NOISE = True   # scale log-likelihood by 1/tau_noise (tau from residual autocorrelation)

# ---- emcee ------------------------------------------------------------------
N_WALKERS      = None        # None -> 4 * ndim (rounded up to even)
INIT_SCALE     = 0.02        # initial walker ball: INIT_SCALE * max(|x0|, 2 % of prior width)
CHUNK          = 2000        # steps between convergence checks
MAX_STEPS      = 40000
TAU_MULT       = 50          # converged when iteration > TAU_MULT * tau for every parameter
N_WORKERS      = None        # None -> all cores; 1 -> no multiprocessing (easier to debug)
N_PRED_DRAWS   = 300         # posterior draws used for the predictive band
SHOW_PLOTS     = False
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

LO  = np.zeros(NDIM)
HI  = np.zeros(NDIM)
GMU = np.full(NDIM, np.nan)     # Gaussian prior mean (nan = uniform)
GSD = np.full(NDIM, np.nan)
for _n in PHYS_NAMES + [NOISE_NAME]:
    LO[IDX[_n]], HI[IDX[_n]] = PRIORS[_n]
if N_PRIOR_LOG10 is not None:
    GMU[IDX["log10_n"]], GSD[IDX["log10_n"]] = N_PRIOR_LOG10
    LO[IDX["log10_n"]] = max(LO[IDX["log10_n"]], N_PRIOR_LOG10[0] - 5 * N_PRIOR_LOG10[1])
    HI[IDX["log10_n"]] = min(HI[IDX["log10_n"]], N_PRIOR_LOG10[0] + 5 * N_PRIOR_LOG10[1])
if MARGINALIZE_SYSTEMATICS:
    for _n in NUIS_NAMES:
        mu, sd, lo, hi = NUISANCE[_n]
        GMU[IDX[_n]], GSD[IDX[_n]] = mu, sd
        LO[IDX[_n]], HI[IDX[_n]] = lo, hi
_GAUSS = ~np.isnan(GSD)

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
# Z lookup (FLYCHK) -- as before, but accepts a z_scale and can stay continuous
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
    """Voigt instrument function on a zero-centred grid with the SAME spacing as wl_m, unit sum."""
    span = wl_m.max() - wl_m.min()
    w = np.linspace(-span / 2, span / 2, wl_m.size)
    v = voigt_profile(w, sigma, gamma)
    return v / v.sum()


def build_v(p):
    """Physical parameter dict from a parameter dict (adds the tied/derived ones)."""
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
    """
    Model spectrum on the data pixels for parameter dict p (physical + nuisance).

    * disp_scale rescales the wavelength axis about the probe line  (pixel -> wavelength error)
    * d_lambda_nm shifts the probe wavelength ONLY                   (laser-line / zero-Doppler error)
      -> these two are independent, unlike the old MC script where they cancelled.
    * sigma_scale / gamma_scale rescale the Voigt instrument function
    * z_scale rescales the table Zbar
    """
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
_CTX = None     # per-process global, set by _init_worker (avoids re-pickling the data every call)


def _init_worker(ctx):
    global _CTX
    _CTX = ctx


def ratio_ok(p):
    if TI_TE_RATIO_BOUNDS is None:
        return True
    lo, hi = TI_TE_RATIO_BOUNDS
    return all(lo <= p[k] / p["T_e_0"] <= hi for k in ("T_i_0", "T_i_1"))


def log_prior(theta):
    if np.any(theta < LO) or np.any(theta > HI):
        return -np.inf
    lp = -0.5 * np.sum(((theta[_GAUSS] - GMU[_GAUSS]) / GSD[_GAUSS]) ** 2)
    return lp


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
    """
    1/tau_int of the residual: tau_int = 1 + 2*sum(rho_k) up to the first non-positive lag.
    Multiplying the log-likelihood by 1/tau_int is the usual 'effective number of independent
    points' correction. It is approximate, and the residual also contains any model mismatch,
    so it errs on the conservative (wider error bar) side.
    """
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
# Starting point: global optimiser (nuisance at nominal)
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
            scale *= 0.5          # shrink the ball if too many starting points are rejected
    return np.array(p0)


def run_sampler(p0, ctx, n_workers, seed):
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
        while sampler.iteration < MAX_STEPS:
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
    """Derived columns for (a random subset of) the samples: Z_Ne, Z_Xe, n_e [1e18 cm^-3], drifts, Ti/Te."""
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
    z = np.abs(np.median(a, 0) - np.median(b, 0)) / (0.5 * (a.std(0) + b.std(0)) + 1e-300)
    return z


# ----------------------------------------------------------------
# Plots
# ----------------------------------------------------------------
def _finish(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


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


def plot_fit(ctx, de_best, flat, map_theta, summ, derived, out_prefix, rng):
    idx = rng.choice(len(flat), min(N_PRED_DRAWS, len(flat)), replace=False)
    curves = np.array([predict(to_params(flat[i]), ctx) for i in idx])
    lo95, lo68, med, hi68, hi95 = np.percentile(curves, [2.5, 15.865, 50, 84.135, 97.5], axis=0)
    wl_nm = ctx["wl_m"] * 1e9

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(wl_nm, lo95, hi95, color="orange", alpha=0.20, label="95 % posterior band")
    ax.fill_between(wl_nm, lo68, hi68, color="orange", alpha=0.40, label="68 % posterior band")
    ax.plot(wl_nm, ctx["data"], "k.", ms=3, label="Data")
    ax.plot(wl_nm, med, "r-", lw=1.6, label="Posterior-median spectrum")
    ax.plot(wl_nm, predict(to_params(map_theta), ctx), "g-", lw=1.0, alpha=0.8, label="MAP spectrum")
    ax.plot(wl_nm, de_best, "b--", lw=1.0, alpha=0.7, label="DE starting fit")
    ax.axvline(ctx["probe_m"] * 1e9, color="red", lw=1)

    def pm(name):
        r = summ.loc[summ.parameter == name].iloc[0]
        return r["median"], r["plus_68"], r["minus_68"]
    ne, nep, nem = pm("n_e_1e18_cm3")
    te, tep, tem = pm("T_e_0")
    t0, t0p, t0m = pm("T_i_0")
    zn = pm("Z_Ne")[0]
    zx = pm("Z_Xe")[0]
    ax.text(0.02, 0.97,
            f"$n_e$ = {ne:.3g} (+{nep:.2g}/-{nem:.2g}) $10^{{18}}$ cm$^{{-3}}$\n"
            f"$T_e$ = {te:.3g} (+{tep:.2g}/-{tem:.2g}) eV\n"
            f"$T_{{i,Ne}}$ = {t0:.3g} (+{t0p:.2g}/-{t0m:.2g}) eV\n"
            f"$Z_{{Ne}}$ = {zn:.2f},  $Z_{{Xe}}$ = {zx:.2f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.8, ec="0.7"))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Skw (normalised)")
    ax.set_title(f"Shot {shot} fiber {fiber}: posterior predictive spectrum")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    _finish(fig, f"{out_prefix}_fit.png")
    return wl_nm, med, lo68, hi68


def plot_contributions(ctx, theta_med, out_prefix):
    """Per-species contributions at the posterior-median parameters (approximate: neglects the change
    in screening when a species is removed). Raw spectra are used so the parts add up to the total."""
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
    ax.legend()
    _finish(fig, f"{out_prefix}_contributions.png")


# ----------------------------------------------------------------
# One full run for one seed
# ----------------------------------------------------------------
def analyse(df, run_id):
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"fiber{fiber}_run{run_id}"
    out_prefix = os.path.join(OUT_DIR, tag)
    fh = logging.FileHandler(f"{out_prefix}.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    try:
        rng = np.random.default_rng(1000 + run_id)

        wl_m = df["wavelength_nm"].to_numpy() * 1e-9
        data = df["intensity"].to_numpy(float)
        data = data / data.max()
        ctx = dict(wl_m=wl_m, data=data, sigma=float(df["sigma"].iloc[0]),
                   gamma=float(df["gamma"].iloc[0]), probe_m=PROBE_M, like_weight=1.0)

        logger.info("=" * 70)
        logger.info(f"Shot {shot} fiber {fiber} run {run_id}   N_data={wl_m.size}   ndim={NDIM}")
        logger.info(f"Systematics marginalised: {MARGINALIZE_SYSTEMATICS}   continuous Z: {CONTINUOUS_Z}")

        # ---- 1. starting point -------------------------------------------------
        best_r2, best = -np.inf, None
        for t in range(DE_TRIES):
            p_de, fit_de, r2 = de_fit(ctx, seed=run_id * 100 + t)
            logger.info(f"  DE try {t + 1}/{DE_TRIES}: R^2 = {r2:.4f}")
            if r2 > best_r2:
                best_r2, best = r2, (p_de, fit_de)
            if r2 >= R2_MIN:
                break
        p_de, fit_de = best
        if best_r2 < R2_MIN:
            logger.warning(f"Best R^2 = {best_r2:.4f} < {R2_MIN}")
            if SKIP_IF_POOR_FIT:
                logger.warning("SKIP_IF_POOR_FIT=True -> skipping this run")
                return None

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
        _init_worker(ctx)            # main process needs ctx too (walker initialisation)
        p0 = init_walkers(x0, nw, rng)
        logger.info(f"  emcee: {nw} walkers, {n_workers} worker(s), chunks of {CHUNK}, max {MAX_STEPS} steps")
        sampler, converged = run_sampler(p0, ctx, n_workers, seed=run_id)

        tau = sampler.get_autocorr_time(tol=0)
        it = sampler.iteration
        tau_max = np.nanmax(tau)
        burn = int(min(2 * tau_max, it // 2)) if np.isfinite(tau_max) else it // 2
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
                sens = [(m, corr.loc[n, m]) for m in NUIS_NAMES if abs(corr.loc[n, m]) > 0.3]
                for m, r in sens:
                    logger.info(f"  systematic sensitivity: {n} <-> {m}: rho = {r:+.2f}")

        np.savez_compressed(f"{out_prefix}_chain.npz", samples=flat, log_prob=lnp,
                            names=np.array(FIT_NAMES), tau=tau, converged=converged)

        # ---- 5. plots + spectrum csv ----------------------------------------------------
        plot_trace(sampler, burn, f"{out_prefix}_trace.png")
        plot_corners(flat, out_prefix)
        wl_nm, med, lo68, hi68 = plot_fit(ctx, fit_de, flat, map_theta, summ, derived, out_prefix, rng)
        plot_contributions(ctx, theta_med, out_prefix)
        pd.DataFrame({"wavelength_nm": wl_nm, "data": data, "de_fit": fit_de,
                      "posterior_median": med, "lo68": lo68, "hi68": hi68}
                     ).to_csv(f"{out_prefix}_fit_spectrum.csv", index=False)
        logger.info(f"Saved everything to {OUT_DIR}  (prefix {tag})")
        return summ
    finally:
        logger.removeHandler(fh)
        fh.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)
    df = pd.read_csv(CSV_PATH)
    for run_id in RUN_IDS:
        analyse(df, run_id)


if __name__ == "__main__":        # required: multiprocessing on macOS uses 'spawn'
    main()
