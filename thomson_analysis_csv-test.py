import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from functools import lru_cache
from lmfit import Parameters, Model as LmModel
from scipy.interpolate import RegularGridInterpolator
from scipy.special import voigt_profile
from plasmapy.diagnostics import thomson
from plasmapy.particles import Particle
import os
import copy

# ---- data ----------------------------------------------------------------
shot=7237
fiber=6
direction='s'
nums=[1,2,3,4,5]
CSV_PATH =f'/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/ts/0{shot}_TS_shot{direction}/0{shot}_TS_shot{direction}_fiber{fiber}.csv'   # <- edit per fiber
PLOT_DIR = f"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/output/{shot}"

# ---- ionization: Z from FLYCHK tables (same as the NN code) ----------------------
# Table file layout (np.loadtxt):
#   row 0 : [dummy, ne_1 ... ne_M]   electron density in cm^-3, strictly increasing
#   col 0 : [Te_1 ... Te_K]          electron temperature in eV, strictly increasing
#   body  : mean ionization Zbar(Te, ne) of the pure element
# Z_used = round(Zbar), clipped to [1, Z_MAX]. H is always H+.
# Outside the table range the lookup is clamped (not extrapolated).
Z_TABLES   = {"Ne": "zforNe.txt", "Xe": "zforXe.txt"}
Z_OVERRIDE = {"Ne": None, "Xe": None}     # int -> fixed charge state, e.g. {"Ne": 9, "Xe": None} if there is no Ne table yet
Z_MAX      = {"Ne": 10, "Xe": 54}

EV_TO_K = 11604.51812155


class ZTable:
    """FLYCHK mean-charge table Zbar(Te [eV], ne [cm^-3]); linear interpolation in (log10 Te, log10 ne), clamped at the edges."""

    def __init__(self, path):
        A = np.loadtxt(path); ne, Te, Z = A[0, 1:], A[1:, 0], A[1:, 1:]
        io, it = np.argsort(ne), np.argsort(Te)
        self.ne, self.Te, self.Z = ne[io], Te[it], Z[it][:, io]
        if self.ne.min() <= 0 or self.Te.min() <= 0:
            raise ValueError(f"{path}: Te and ne must be > 0")
        if np.any(np.diff(self.ne) <= 0) or np.any(np.diff(self.Te) <= 0):
            raise ValueError(f"{path}: duplicate Te or ne values")
        self.interp = RegularGridInterpolator((np.log10(self.Te), np.log10(self.ne)), self.Z, method="linear")

    def __call__(self, Te, ne_cm3):
        x = np.log10(np.clip(Te, self.Te[0], self.Te[-1]))
        y = np.log10(np.clip(ne_cm3, self.ne[0], self.ne[-1]))
        return self.interp(np.stack([x, y], -1))


def _load_z_sources():
    src = {}
    for el in ("Ne", "Xe"):
        if Z_OVERRIDE.get(el) is not None:
            src[el] = int(Z_OVERRIDE[el])
            print(f"Z_{el} fixed to {src[el]}+ (Z_OVERRIDE)")
        else:
            if not os.path.exists(Z_TABLES[el]):
                raise SystemExit(f"Z lookup table missing: {Z_TABLES[el]}\n"
                                 f"Provide it, or set Z_OVERRIDE['{el}'] to a fixed charge state.")
            src[el] = ZTable(Z_TABLES[el])
    return src

Z_SRC = _load_z_sources()


def get_Z(Te_eV, n_m3):
    """(Z_Ne, Z_Xe) from the lookup at the given Te [eV] and n [m^-3]."""
    out = []
    for el in ("Ne", "Xe"):
        s = Z_SRC[el]
        if isinstance(s, int):
            out.append(s)
        else:
            zb = np.asarray(s(np.atleast_1d(float(Te_eV)), np.atleast_1d(float(n_m3)) * 1e-6)).ravel()[0]
            out.append(int(np.clip(np.rint(zb), 1, Z_MAX[el])))
    return tuple(out)


@lru_cache(maxsize=None)
def ion_mass(el, z):
    sym = {"Ne": "Ne-20", "Xe": "Xe-132"}[el]
    return Particle(f"{sym} {z}+").mass.to(u.kg).value

H_MASS = Particle("H+").mass.to(u.kg).value

# ---- scattering geometry --------------------------------------------------
PROBE_M = (526.5 * u.nm).to(u.m).value                    # probe laser wavelength
probe_vec = np.array([1, 0, 0])                           # probe beam direction
scatter_vec = np.array([np.cos(np.deg2rad(90)),           # collection direction;
                         np.sin(np.deg2rad(90)), 0])      # angle to probe_vec = scattering angle

# Thomson scattering only measures velocity along k = k_scatter - k_probe
k_vec = scatter_vec - probe_vec
k_hat = k_vec / np.linalg.norm(k_vec)

NOTCH = None   # e.g. (np.array([526.25, 526.6]) * u.nm).to(u.m).value  # optional: zero out a wavelength range

ELEMENT_LABELS = lambda z_ne, z_xe: [f"Ne-20 {z_ne}+", "H+", f"Xe-132 {z_xe}+"]


# ---- forward model (Z is passed in, so it can come from the lookup on every call) -------------
def instr_array(wl_m, sigma, gamma):
    """Voigt instrument function on a zero-centred grid, normalised to unit sum (same as plasmapy builds it)."""
    span = wl_m.max() - wl_m.min()
    w = np.linspace(-span / 2, span / 2, wl_m.size)
    v = voigt_profile(w, sigma, gamma)
    return v / v.sum()


def forward(wl_m, v, z_ne, z_xe, instr, ifr=None):
    """v: dict of physical params (n, T_e_0, T_i_0..2, ion_speed_0..2, electron_speed_0, ifract_0..2). Max-normalised."""
    ifract = np.array([v['ifract_0'], v['ifract_1'], v['ifract_2']]) if ifr is None else np.asarray(ifr, float)
    T_i = np.array([v['T_i_0'], v['T_i_1'], v['T_i_2']]) * EV_TO_K
    z = np.array([z_ne, 1.0, z_xe], float)
    m = np.array([ion_mass("Ne", z_ne), H_MASS, ion_mass("Xe", z_xe)])
    ion_vel = np.array([v['ion_speed_0'], v['ion_speed_1'], v['ion_speed_2']])[:, None] * k_hat[None, :]
    el_vel = (v['electron_speed_0'] * k_hat)[None, :]
    kw = {} if NOTCH is None else {"notch": NOTCH}
    _, S = thomson.spectral_density_lite(
        wl_m, PROBE_M, v['n'], np.array([v['T_e_0'] * EV_TO_K]), T_i, np.array([1.0]),
        ifract, z, m, el_vel, ion_vel, probe_vec, scatter_vec, instr_func_arr=instr, **kw)
    return S / S.max()


def make_model(wl_m, sigma, gamma):
    """lmfit model; Z_Ne and Z_Xe are looked up from (T_e_0, n) on EVERY evaluation."""
    instr = instr_array(wl_m, sigma, gamma)

    def f(wavelengths, n, T_e_0, T_i_0, T_i_1, T_i_2, ion_speed_0, ion_speed_1, ion_speed_2,
          electron_speed_0, ifract_0, ifract_1, ifract_2):
        v = dict(n=n, T_e_0=T_e_0, T_i_0=T_i_0, T_i_1=T_i_1, T_i_2=T_i_2,
                 ion_speed_0=ion_speed_0, ion_speed_1=ion_speed_1, ion_speed_2=ion_speed_2,
                 electron_speed_0=electron_speed_0, ifract_0=ifract_0, ifract_1=ifract_1, ifract_2=ifract_2)
        z_ne, z_xe = get_Z(T_e_0, n)
        return forward(wavelengths, v, z_ne, z_xe, instr)

    return LmModel(f, independent_vars=["wavelengths"]), instr


ion_names = ['Ne', 'H', 'Xe']
n_ions = len(ion_names)

for num in nums:
    df = pd.read_csv(CSV_PATH)
    wavelengths = df['wavelength_nm'].to_numpy() * u.nm
    wl_m = wavelengths.to(u.m).value
    skw = df['intensity'].to_numpy()
    skw = skw / skw.max()
    sigma = df['sigma'].iloc[0]   # instrument Voigt params come from the CSV itself
    gamma = df['gamma'].iloc[0]

    model, instr = make_model(wl_m, sigma, gamma)

    # ---- params (fit variables) ----------------------------------------------
    params = Parameters()
    params.add('n', value=5e17 * 1e6, min=1e23, max=1e27, vary=True)   # total electron density, m^-3

    # --- per-ion-species: T_i_<i>, ion_speed_<i> ---
    params.add('T_i_0', value=150, min=10, max=5000, vary=True)        # eV
    params.add('T_i_1', value=150, min=1, max=500, vary=True)        # eV
    params.add('T_i_2', expr='T_i_1')#value=100, min=0.5, max=500, vary=True)        # eV
    params.add('ion_speed_0', value=0, min=-5e5, max=5e5, vary=True)   # m/s along k_hat
    params.add('ion_speed_1', value=0, min=-5e5, max=5e5, vary=True)   # m/s along k_hat
    params.add('ion_speed_2', value=0, min=-5e5, max=5e5, vary=True)   # m/s along k_hat

    params.add('ifract_0', value=0.4, min=0, max=0.6, vary=True)
    params.add('ifract_1', expr='(1 - ifract_0) * 1/(1 + 0.0065)')
    params.add('ifract_2', expr='(1 - ifract_0) * 0.0065/(1 + 0.0065)')

    # --- electrons: T_e_<e>, electron_speed_<e> ---
    params.add('T_e_0', value=50, min=0.5, max=800, vary=True)             # eV
    params.add('electron_speed_0', value=0, min=-5e5, max=5e5, vary=True)   # m/s along k_hat

    # ---- fit (retry until R² > 0.98) -----------------------------------------
    r2 = -1
    tries=0

    while r2 < 0.98 and tries<4:

        result = model.fit(
            skw,
            params=copy.deepcopy(params),
            wavelengths=wl_m,
            method="differential_evolution"
        )

        r2 = 1 - result.residual.var() / np.var(skw)
        tries+=1
        print(r2)
    if r2 <0.98:
        continue

    # ---- best-fit values and the charge states the lookup gives for them --------
    p = result.params
    v_best = {k: p[k].value for k in ['n', 'T_e_0', 'T_i_0', 'T_i_1', 'T_i_2', 'ion_speed_0', 'ion_speed_1',
                                       'ion_speed_2', 'electron_speed_0', 'ifract_0', 'ifract_1', 'ifract_2']}
    z_ne, z_xe = get_Z(v_best['T_e_0'], v_best['n'])
    ions = ELEMENT_LABELS(z_ne, z_xe)

    # ---- plot -------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(wavelengths.value, skw, label="Data")
    ax.plot(wavelengths.value, result.best_fit, label="Best fit")
    ax.axvline((PROBE_M * u.m).to(u.nm).value, color="red")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Skw")
    ax.set_title(f"{ions[0]}, {ions[2]} (Z from lookup)", fontsize=9)
    ax.legend()

    # ---- results -----------------------------------------------------------------
    def _fmt(val, err, unit):
        return f"{val:.3g} ± {err:.3g} {unit}" if err is not None else f"{val:.3g} {unit}"

    def _fmt2(val, err, unit):
        return f"{val:.3g} ± {err:.3g} {unit}" if err is not None else f"{val:.3f} {unit}"

    lines = []
    lines.append(f"n           = {_fmt(p['n'].value * 1e-6 / 1e18, (p['n'].stderr * 1e-6 / 1e18) if p['n'].stderr else None, '×10¹⁸ cm⁻³')}")
    for i in range(n_ions):
        lines.append(f"T_i_{i}        = {_fmt2(p[f'T_i_{i}'].value, p[f'T_i_{i}'].stderr, 'eV')}")
    lines.append(f"T_e_0       = {_fmt(p['T_e_0'].value, p['T_e_0'].stderr, 'eV')}")
    for i in range(n_ions):
        lines.append(f"ifract_{i}     = {_fmt(p[f'ifract_{i}'].value, p[f'ifract_{i}'].stderr, '')}")
    for i in range(n_ions):
        v, e = p[f'ion_speed_{i}'].value, p[f'ion_speed_{i}'].stderr
        lines.append(f"ion_speed_{i}  = {_fmt(v / 1e3, e / 1e3 if e else None, 'km/s')}")

    v, e = p['electron_speed_0'].value, p['electron_speed_0'].stderr
    lines.append(f"e_speed_0   = {_fmt(v / 1e3, e / 1e3 if e else None, 'km/s')}")
    lines.append(f"Z_Ne, Z_Xe  = {z_ne}+, {z_xe}+  (lookup at best-fit Te, n)")
    lines.append(f"red. χ²     = {result.redchi:.4g}")
    lines.append(f"R²          = {r2:.4f}")

    print("\n".join(lines))

    # ---- save only good fits ---------------------------------------------------
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig.savefig(f"{PLOT_DIR}/fiber{fiber}_fit_{num}.png", dpi=300, bbox_inches="tight")

    plt.show()

    fit_df = pd.DataFrame({
        "wavelength_nm": wavelengths.value,
        "data": skw,
        "best_fit": result.best_fit
    })

    fit_df.to_csv(f"{PLOT_DIR}/fiber{fiber}_{num}_fit_spectrum.csv", index=False)

    # ---- contribution plot -----------------------------------------------------
    plt.figure()
    plt.plot(wavelengths.value, result.best_fit, 'k', lw=2, label='Total fit')

    Z = [z_ne, 1, z_xe]          # charge states from the lookup (no longer hard-coded)
    ifr_best = np.array([v_best[f'ifract_{j}'] for j in range(n_ions)])

    weights = ifr_best * np.array(Z, float)
    weights /= weights.sum()

    for i in range(n_ions):
        spec = forward(wl_m, v_best, z_ne, z_xe, instr, ifr=np.eye(n_ions)[i])
        spec *= weights[i]
        plt.plot(wavelengths.value, spec, label=ions[i])

    plt.legend()
    plt.savefig(f"{PLOT_DIR}/fiber{fiber}_contributions_{num}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

    # ---- save CSV --------------------------------------------------------------
    out = []
    for name, par in result.params.items():
        out.append({
            "parameter": name,
            "value": par.value,
            "stderr": par.stderr
        })
    out.append({"parameter": "Z_Ne", "value": z_ne, "stderr": np.nan})
    out.append({"parameter": "Z_Xe", "value": z_xe, "stderr": np.nan})

    df_out = pd.DataFrame(out)

    OUTPUT_PATH = f"{PLOT_DIR}/fiber{fiber}_{num}.csv"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved results to {OUTPUT_PATH}")
