import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters

from plasmapy.diagnostics import thomson

df=pd.read_csv('/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/fiber13.csv')

iaw_skw = df['intensity'].to_numpy()
iaw_skw /= iaw_skw.max()
iaw_wavelengths = df['wavelength_nm'].to_numpy()*u.nm

probe_wavelength = 526.5*u.nm
probe_vec = np.array([1,0,0])
scattering_angle = np.deg2rad(90)
scatter_vec = scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])

ions = ['H+']
ion_vdir = np.array([[1,0,0]])
electron_vdir = np.array([[1,0,0]])

params = Parameters()

params.add('n',                value=1e18 * 1e6,      min=1e19,  max=1e27,   vary=True)   # m⁻³
params.add('T_e_0',            value=50,              min=1.0,   max=2000.0, vary=True)   # eV
params.add('T_i_0',            value=50,              min=0.5,   max=500.0,  vary=True)   # eV
params.add('ion_speed_0',      value=0,               min=-5e5,  max=5e5,    vary=True)   # m/s
params.add('electron_speed_0', value=0,               min=-5e5,  max=5e5,    vary=True)   # m/s

settings = {}

settings["probe_wavelength"] = probe_wavelength.to(u.m).value
settings["probe_vec"] = probe_vec
settings["scatter_vec"] = scatter_vec
settings["ions"] = ions
settings["ion_vdir"] = ion_vdir
settings["electron_vdir"]  = electron_vdir

iaw_model = thomson.spectral_density_model(
    iaw_wavelengths.to(u.m).value,
    settings,
    params,
)

iaw_result = iaw_model.fit(
    iaw_skw,
    params=params,
    wavelengths=iaw_wavelengths.to(u.m).value,
    method="differential_evolution",
)

# Extract the best fit curve
best_fit_skw = iaw_result.best_fit

# Plot
fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Skw")
ax.axvline(x=probe_wavelength.value, color="red", label="Probe wavelength")

ax.set_xlim(523, 530)

ax.plot(iaw_wavelengths.value, iaw_skw, label="Data")
ax.plot(iaw_wavelengths.value, best_fit_skw, label="Best-fit")
ax.legend(loc="upper right")
plt.show()

'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from lmfit import Parameters
from plasmapy.diagnostics.thomson import spectral_density_model

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/fiber13.csv')

iaw_skw         = df['intensity'].to_numpy().astype(float)
iaw_wavelengths = df['wavelength_nm'].to_numpy()  # nm, plain float64

# ── Parameters (tweak these to match your experiment) ─────────────────────────
probe_wavelength     = 526.5        # nm
iaw_window_nm        = 2.0          # ± window around probe to fit
ne_guess             = 1e18         # cm⁻³
T_e_guess            = 100.0        # eV
T_i_guess            = 20.0         # eV
v_flow_guess         = 0.0          # m/s
ion_species          = 'H+'
scattering_angle_deg = 90

# ── Geometry ───────────────────────────────────────────────────────────────────
angle       = np.deg2rad(scattering_angle_deg)
probe_vec   = np.array([1, 0, 0])
scatter_vec = np.array([np.cos(angle), np.sin(angle), 0])
k_vec       = scatter_vec - probe_vec
k_hat       = k_vec / np.linalg.norm(k_vec)

# ── Window + normalise ─────────────────────────────────────────────────────────
mask    = np.abs(iaw_wavelengths - probe_wavelength) <= iaw_window_nm
wl_nm   = iaw_wavelengths[mask]
wl_m    = wl_nm * 1e-9              # metres, plain float64
sp_data = iaw_skw[mask]

if sp_data.max() <= 0:
    raise ValueError("Spectrum in the fit window is non-positive — check your data.")

sp_norm = sp_data / sp_data.max()   # normalise to [0, 1]

# ── PlasmaPy settings dict (no Quantities — bare floats/arrays only) ───────────
settings = {
    'probe_wavelength' : probe_wavelength * 1e-9,   # metres
    'probe_vec'        : probe_vec,
    'scatter_vec'      : scatter_vec,
    'ions'             : [ion_species],
    'ion_vdir'         : k_hat.reshape(1, 3),
    'electron_vdir'    : k_hat.reshape(1, 3),
}

# ── lmfit Parameters ───────────────────────────────────────────────────────────
params = Parameters()
params.add('n',                value=ne_guess * 1e6,  min=1e19,  max=1e27,   vary=True)   # m⁻³
params.add('T_e_0',            value=T_e_guess,       min=1.0,   max=2000.0, vary=True)   # eV
params.add('T_i_0',            value=T_i_guess,       min=0.5,   max=500.0,  vary=True)   # eV
params.add('ion_speed_0',      value=v_flow_guess,    min=-5e5,  max=5e5,    vary=True)   # m/s
params.add('electron_speed_0', value=v_flow_guess,    min=-5e5,  max=5e5,    vary=True)   # m/s
params.add('background',       value=0.0,             min=-0.2,  max=0.2,    vary=True)

# ── Build model and fit ────────────────────────────────────────────────────────
model  = spectral_density_model(wl_m * u.m, settings, params)
result = model.fit(
    sp_norm,
    params=params,
    wavelengths=wl_m,
    method='differential_evolution',
)

# ── Extract results ────────────────────────────────────────────────────────────
T_i    = result.params['T_i_0'].value
T_e    = result.params['T_e_0'].value
ne     = result.params['n'].value * 1e-6        # back to cm⁻³
v_flow = result.params['ion_speed_0'].value

T_i_err    = result.params['T_i_0'].stderr
T_e_err    = result.params['T_e_0'].stderr
ne_err     = result.params['n'].stderr * 1e-6 if result.params['n'].stderr else None
v_flow_err = result.params['ion_speed_0'].stderr

def _fmt(val, err, unit):
    return f"{val:.2f} ± {err:.2f} {unit}" if err is not None else f"{val:.2f} {unit}"

print(
    f"T_i    = {_fmt(T_i, T_i_err, 'eV')}\n"
    f"T_e    = {_fmt(T_e, T_e_err, 'eV')}\n"
    f"ne     = {_fmt(ne/1e18, ne_err/1e18 if ne_err else None, '×10¹⁸ cm⁻³')}\n"
    f"v_flow = {_fmt(v_flow/1e3, v_flow_err/1e3 if v_flow_err else None, 'km/s')}\n"
    f"red. χ² = {result.redchi:.4f}"
)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wl_nm, sp_norm,         'k.', ms=3,  label='Data (normalised)')
ax.plot(wl_nm, result.best_fit, 'r-', lw=1.5,
        label=(f"$T_i$={T_i:.1f} eV,  $T_e$={T_e:.1f} eV\n"
               f"$n_e$={ne/1e18:.2f}×10¹⁸ cm⁻³,  "
               f"$v$={v_flow/1e3:.1f} km/s"))
ax.axhline(0,                color='grey', ls=':',  lw=0.8)
ax.axvline(probe_wavelength, color='grey', ls='--', lw=0.8, label='Probe λ')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Normalised intensity')
ax.set_title('IAW fit — fiber13')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
'''
