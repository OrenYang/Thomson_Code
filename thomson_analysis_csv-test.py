import pickle
from tkinter import filedialog, Tk
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters
from scipy.special import voigt_profile

from plasmapy.diagnostics import thomson

class Thomson:
    def __init__(self, pkl_file=None):
        if pkl_file is None:
            pkl_file = filedialog.askopenfilename(title="Select a .pkl file")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(data)

DEFAULT_CONFIG = {
    # Experimental setup
    "probe_wavelength":     526.5 * u.nm,
    "probe_vec":            np.array([1, 0, 0]),
    "scattering_angle":     np.deg2rad(90),
    "method":               "differential_evolution",
    "sigma":                1.3e-11,
    "gamma":                1.1e-11,

    "ions":                 ['Ne-20 +8', 'Ne-20 +2', 'Ne-20 0+'],
    "ion_vdir":             np.array([[1,0,0],[1,0,0],[1,0,0]]),
    "electron_vdir":        np.array([[1,0,0]]),

    # ifract
    "ifract":               [0.33, 0.33, 0.33],
    "ifract_vary":          [True, True], #last species will vary to sum to 1

    # Ion temperature
    "T_ion":                [50, 10, 1],
    "T_ion_min":            [0.5, 0.5, 0.5],
    "T_ion_max":            [3000, 3000, 3000],
    "T_ion_vary":           [True, True, True],

    # Ion speed
    "ion_speed":            [0, 0, 0],
    "ion_speed_min":        [-5e5, -5e5, -5e5],
    "ion_speed_max":        [5e5,  5e5,  5e5],
    "ion_speed_vary":       [True, True, True],

    # Electron temperature
    "T_e":                  50,
    "T_e_min":              0.5,
    "T_e_max":              3000.0,
    "T_e_vary":             True,

    # Electron speed
    "e_speed":              0,
    "e_speed_min":          -5e5,
    "e_speed_max":          5e5,
    "e_speed_vary":         True,

    # Density
    "n":                    1e18 * 1e6,
    "n_min":                1e22,
    "n_max":                1e27,
    "n_vary":               True,
}

def _make_instr_func(sigma,gamma):
    def instrument_function(wavelengths):
        w = wavelengths.to(u.m).value
        instr = voigt_profile(w, sigma, gamma)
        return instr / instr.max()
    return instrument_function

def fit_iaw(iaw_skw, iaw_wavelengths, config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    scatter_vec = np.array([np.cos(cfg['scattering_angle']), np.sin(cfg['scattering_angle']), 0])
    # Convert ifract initial values -> sfract so ifract can sum to 1 for arbitrary number of ions
    sfracts = []
    remaining = 1.0
    for i in range(len(cfg['ions']) - 1):
        s = cfg['ifract'][i] / remaining
        sfracts.append(s)
        remaining -= cfg['ifract'][i]


    params = Parameters()

    params.add('n',                       value=cfg['n'],               min=cfg['n_min'],               max=cfg['n_max'],               vary=cfg['n_vary'])
    for i in range(len(cfg['ions'])):
        params.add(f'T_i_{i}',            value=cfg['T_ion'][i],        min=cfg['T_ion_min'][i],        max=cfg['T_ion_max'][i],        vary=cfg['T_ion_vary'][i])
        params.add(f'ion_speed_{i}',      value=cfg['ion_speed'][i],    min=cfg['ion_speed_min'][i],    max=cfg['ion_speed_max'][i],    vary=cfg['ion_speed_vary'][i])
    # Add sfract as the free parameters
    for i in range(len(cfg['ions']) - 1):
        params.add(f'sfract_{i}',         value=sfracts[i],             min=0,                          max=1,                          vary=cfg['ifract_vary'][i])
    # Derive ifract from sfract
    for i in range(len(cfg['ions']) - 1):
        if i == 0:
            expr = 'sfract_0'
        else:
            remaining = ' * '.join([f'(1 - sfract_{j})' for j in range(i)])
            expr = f'sfract_{i} * {remaining}'
        params.add(f'ifract_{i}', expr=expr)
    # Last ifract gets the remainder
    params.add(f'ifract_{len(cfg['ions'])-1}', expr='1 - ' + ' - '.join([f'ifract_{j}' for j in range(len(cfg['ions'])-1)]))
    params.add('T_e_0',                   value=cfg['T_e'],             min=cfg['T_e_min'],             max=cfg['T_e_max'],             vary=cfg['T_e_vary'])
    params.add('electron_speed_0',        value=cfg['e_speed'],         min=cfg['e_speed_min'],         max=cfg['e_speed_max'],         vary=cfg['e_speed_vary'])


    settings = {}

    settings["probe_wavelength"] = cfg['probe_wavelength'].to(u.m).value
    settings["probe_vec"] = cfg['probe_vec']
    settings["scatter_vec"] = scatter_vec
    settings["ions"] = cfg['ions']
    settings["ion_vdir"] = cfg['ion_vdir']
    settings["electron_vdir"]  = cfg['electron_vdir']
    settings["instr_func"] = _make_instr_func(cfg['sigma'],cfg['gamma'])

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

    ############## PLOT ##########################
    # Extract the best fit curve
    best_fit_skw = iaw_result.best_fit

    fig, ax = plt.subplots(ncols=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Skw")
    ax.axvline(x=cfg['probe_wavelength'].value, color="red", label="Probe wavelength")

    ax.set_xlim(523, 530)

    ax.plot(iaw_wavelengths.value, iaw_skw, label="Data")
    ax.plot(iaw_wavelengths.value, best_fit_skw, label="Best-fit")
    ax.legend(loc="upper right")
    plt.show()

    return iaw_result

df=pd.read_csv('/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/fiber13.csv')

iaw_skw = df['intensity'].to_numpy()
iaw_skw /= iaw_skw.max()
iaw_wavelengths = df['wavelength_nm'].to_numpy()*u.nm

ions = ['Ne-20 +8', 'Ne-20 +2', 'Ne-20 0+']
ion_speed = [0, 0, 0]
ifract = [0.33,0.33,0.33]
T_ion = [50, 10, 1]
T_ion_max = [3000, 300, 30]
ion_vdir = np.array([[1,0,0],
                     [1,0,0],
                     [1,0,0]])
electron_vdir = np.array([[1,0,0]])

iaw_result = fit_iaw(iaw_skw, iaw_wavelengths)

def _fmt(val, err, unit):
    return f"{val:.2f} ± {err:.2f} {unit}" if err is not None else f"{val:.2f} {unit}"

T_i_0       = iaw_result.params['T_i_0'].value
T_i_1       = iaw_result.params['T_i_1'].value
T_i_2       = iaw_result.params['T_i_2'].value
T_e_0       = iaw_result.params['T_e_0'].value
ne          = iaw_result.params['n'].value * 1e-6
ifract_0    = iaw_result.params['ifract_0'].value
ifract_1    = iaw_result.params['ifract_1'].value
ifract_2    = iaw_result.params['ifract_2'].value
ion_speed_0 = iaw_result.params['ion_speed_0'].value
ion_speed_1 = iaw_result.params['ion_speed_1'].value
ion_speed_2 = iaw_result.params['ion_speed_2'].value
e_speed_0   = iaw_result.params['electron_speed_0'].value

T_i_0_err       = iaw_result.params['T_i_0'].stderr
T_i_1_err       = iaw_result.params['T_i_1'].stderr
T_i_2_err       = iaw_result.params['T_i_2'].stderr
T_e_0_err       = iaw_result.params['T_e_0'].stderr
ne_err          = iaw_result.params['n'].stderr * 1e-6 if iaw_result.params['n'].stderr else None
ifract_0_err    = iaw_result.params['ifract_0'].stderr
ifract_1_err    = iaw_result.params['ifract_1'].stderr
ifract_2_err    = iaw_result.params['ifract_2'].stderr
ion_speed_0_err = iaw_result.params['ion_speed_0'].stderr
ion_speed_1_err = iaw_result.params['ion_speed_1'].stderr
ion_speed_2_err = iaw_result.params['ion_speed_2'].stderr
e_speed_0_err   = iaw_result.params['electron_speed_0'].stderr

print(
    f"T_i_0       = {_fmt(T_i_0,            T_i_0_err,                                    'eV')}\n"
    f"T_i_1       = {_fmt(T_i_1,            T_i_1_err,                                    'eV')}\n"
    f"T_i_2       = {_fmt(T_i_2,            T_i_2_err,                                    'eV')}\n"
    f"T_e_0       = {_fmt(T_e_0,            T_e_0_err,                                    'eV')}\n"
    f"ne          = {_fmt(ne/1e18,          ne_err/1e18 if ne_err else None,              '×10¹⁸ cm⁻³')}\n"
    f"ifract_0    = {_fmt(ifract_0,         ifract_0_err,                                 '')}\n"
    f"ifract_1    = {_fmt(ifract_1,         ifract_1_err,                                 '')}\n"
    f"ifract_2    = {_fmt(ifract_2,         ifract_2_err,                                 '')}\n"
    f"ion_speed_0 = {_fmt(ion_speed_0/1e3, ion_speed_0_err/1e3 if ion_speed_0_err else None, 'km/s')}\n"
    f"ion_speed_1 = {_fmt(ion_speed_1/1e3, ion_speed_1_err/1e3 if ion_speed_1_err else None, 'km/s')}\n"
    f"ion_speed_2 = {_fmt(ion_speed_2/1e3, ion_speed_2_err/1e3 if ion_speed_2_err else None, 'km/s')}\n"
    f"e_speed_0   = {_fmt(e_speed_0/1e3,   e_speed_0_err/1e3  if e_speed_0_err  else None,  'km/s')}\n"
    f"red. χ²     = {iaw_result.redchi:.4f}"
)
