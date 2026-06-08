import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters

from plasmapy.diagnostics import thomson

# Generate theoretical spectrum
probe_wavelength = 532 * u.nm
epw_wavelengths = (
    np.linspace(probe_wavelength.value - 30, probe_wavelength.value + 30, num=500)
    * u.nm
)
iaw_wavelengths = (
    np.linspace(probe_wavelength.value - 3, probe_wavelength.value + 3, num=500) * u.nm
)

probe_vec = np.array([1, 0, 0])
scattering_angle = np.deg2rad(63)
scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])

n = 2e17 * u.cm**-3
ions = ["H+", "C-12 5+"]
T_e = 10 * u.eV
T_i = np.array([20, 50]) * u.eV
electron_vel = np.array([[0, 0, 0]]) * u.km / u.s
ion_vel = np.array([[0, 0, 0], [200, 0, 0]]) * u.km / u.s
ifract = [0.3, 0.7]

alpha, epw_skw = thomson.spectral_density(
    epw_wavelengths,
    probe_wavelength,
    n,
    T_e=T_e,
    T_i=T_i,
    ions=ions,
    ifract=ifract,
    electron_vel=electron_vel,
    ion_vel=ion_vel,
    probe_vec=probe_vec,
    scatter_vec=scatter_vec,
)

alpha, iaw_skw = thomson.spectral_density(
    iaw_wavelengths,
    probe_wavelength,
    n,
    T_e=T_e,
    T_i=T_i,
    ions=ions,
    ifract=ifract,
    electron_vel=electron_vel,
    ion_vel=ion_vel,
    probe_vec=probe_vec,
    scatter_vec=scatter_vec,
)

# PLOTTING
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
fig.subplots_adjust(wspace=0.2)

for a in ax:
    a.set_xlabel("Wavelength (nm)")
    a.set_ylabel("Skw")
    a.axvline(x=probe_wavelength.value, color="red", label="Probe wavelength")

ax[0].set_xlim(520, 545)
ax[0].set_ylim(0, 3e-13)
ax[0].set_title("Electron Plasma Wave")
ax[0].plot(epw_wavelengths.value, epw_skw)
ax[0].legend()

ax[1].set_xlim(530, 534)
ax[1].set_title("Ion Acoustic Wave")
ax[1].plot(iaw_wavelengths.value, iaw_skw)
ax[1].legend();

epw_skw *= 1 + np.random.normal(loc=0, scale=0.1, size=epw_wavelengths.size)

iaw_skw *= 1 + np.random.normal(loc=0, scale=0.1, size=iaw_wavelengths.size)

notch_range = (531, 533)
x0 = np.argmin(np.abs(epw_wavelengths.value - notch_range[0]))
x1 = np.argmin(np.abs(epw_wavelengths.value - notch_range[1]))
epw_skw[x0:x1] = np.nan

epw_skw = np.delete(epw_skw, slice(x0,x1,None))
epw_wavelengths = np.delete(epw_wavelengths, slice(x0,x1,None))

epw_skw = epw_skw.value
epw_skw *= 1 / np.nanmax(epw_skw)
iaw_skw = iaw_skw.value
iaw_skw *= 1 / np.nanmax(iaw_skw)

# Plot again
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
fig.subplots_adjust(wspace=0.2)

for a in ax:
    a.set_xlabel("Wavelength (nm)")
    a.set_ylabel("Skw")
    a.axvline(x=probe_wavelength.value, color="red", label="Probe wavelength")

ax[0].set_xlim(520, 545)
ax[0].set_title("Electron Plasma Wave")
ax[0].plot(epw_wavelengths.value, epw_skw)
ax[0].legend()

ax[1].set_xlim(531, 533)
ax[1].set_title("Ion Acoustic Wave")
ax[1].plot(iaw_wavelengths.value, iaw_skw)
ax[1].legend();

params = Parameters()
params.add(
    "n", value=4e17 * 1e6, vary=True, min=5e16 * 1e6, max=1e18 * 1e6
)  # Converting cm^-3 to m^-3
params.add("T_e_0", value=5, vary=True, min=0.5, max=25)
params.add("T_i_0", value=5, vary=False)
params.add("T_i_1", value=10, vary=False)
params.add("ifract_0", value=0.8, vary=False)
params.add("ifract_1", value=0.2, vary=False)
params.add("ion_speed_0", value=0, vary=False)
params.add("ion_speed_1", value=0, vary=False)

settings = {}
settings["probe_wavelength"] = probe_wavelength.to(u.m).value
settings["probe_vec"] = probe_vec
settings["scatter_vec"] = scatter_vec
settings["ions"] = ions
settings["ion_vdir"] = np.array([[1, 0, 0], [1, 0, 0]])

epw_model = thomson.spectral_density_model(
    epw_wavelengths.to(u.m).value, settings, params
)

fit_kws = {}
epw_result = epw_model.fit(
    epw_skw,
    params=params,
    wavelengths=epw_wavelengths.to(u.m).value,
    method="differential_evolution",
    fit_kws=fit_kws,
)

# Print some of the results compared to the true values
answers = {"n": 2e17, "T_e_0": 10}
for key, ans in answers.items():
    print(f"{key}: {epw_result.best_values[key]:.1e} (true value {ans:.1e})")

print(f"Number of fit iterations:{epw_result.nfev}")
print(f"Reduced Chisquared:{epw_result.redchi:.4f}")

# Extract the best fit curve by evaluating the model at the final parameters
n_fit = epw_result.values["n"]
Te_0_fit = epw_result.values["T_e_0"]

# Extract the best fit curve
best_fit_skw = epw_result.best_fit

# Get all the non-nan indices (the best_fit_skw just omits these values)
not_nan = np.argwhere(np.logical_not(np.isnan(epw_skw)))

# Plot
fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Skw")
ax.axvline(x=probe_wavelength.value, color="red", label="Probe wavelength")

ax.set_xlim(520, 545)

ax.plot(epw_wavelengths.value, epw_skw, label="Data")
ax.plot(epw_wavelengths.value[not_nan], best_fit_skw, label="Best-fit")
ax.legend(loc="upper right");

settings = {}
settings["probe_wavelength"] = probe_wavelength.to(u.m).value
settings["probe_vec"] = probe_vec
settings["scatter_vec"] = scatter_vec
settings["ions"] = ions
settings["ion_vdir"] = np.array([[1, 0, 0], [1, 0, 0]])

params = Parameters()
params.add("n", value=n_fit, vary=False)
params.add("T_e_0", value=Te_0_fit, vary=False)
params.add("T_i_0", value=10, vary=True, min=5, max=60)
params.add("T_i_1", value=10, vary=True, min=5, max=60)
params.add("ifract_0", value=0.5, vary=True, min=0.2, max=0.8)
params.add("ifract_1", value=0.5, vary=True, min=0.2, max=0.8, expr="1.0 - ifract_0")
params.add("ion_speed_0", value=0, vary=False)
params.add("ion_speed_1", value=0, vary=True, min=0, max=1e6)

iaw_model = thomson.spectral_density_model(
    iaw_wavelengths.to(u.m).value, settings, params
)

print(iaw_skw.shape)

iaw_result = iaw_model.fit(
    iaw_skw,
    params=params,
    wavelengths=iaw_wavelengths.to(u.m).value,
    method="differential_evolution",
)

# Print some of the results compared to the true values
answers = {
    "T_i_0": 20,
    "T_i_1": 50,
    "ifract_0": 0.3,
    "ifract_1": 0.7,
    "ion_speed_1": 2e5,
}
for key, ans in answers.items():
    print(f"{key}: {iaw_result.best_values[key]:.1f} (true value {ans:.1f})")

print(f"Number of fit iterations:{iaw_result.nfev:.1f}")
print(f"Reduced Chisquared:{iaw_result.redchi:.4f}")

# Extract the best fit curve
best_fit_skw = iaw_result.best_fit

# Plot
fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Skw")
ax.axvline(x=probe_wavelength.value, color="red", label="Probe wavelength")

ax.set_xlim(531, 533)

ax.plot(iaw_wavelengths.value, iaw_skw, label="Data")
ax.plot(iaw_wavelengths.value, best_fit_skw, label="Best-fit")
ax.legend(loc="upper right");
