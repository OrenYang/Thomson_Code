import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from lmfit import Parameters
from scipy.special import voigt_profile
from plasmapy.diagnostics import thomson
import os
import copy

# ---- data ----------------------------------------------------------------
shot=7232
fiber=6
nums=[1,2,3,4,5]
CSV_PATH =f'/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts/0{shot}_TS_shots/0{shot}_TS_shots_fiber{fiber}.csv'   # <- edit per fiber

for num in nums:
    df = pd.read_csv(CSV_PATH)
    wavelengths = df['wavelength_nm'].to_numpy() * u.nm
    skw = df['intensity'].to_numpy()
    skw = skw / skw.max()
    sigma = df['sigma'].iloc[0]   # instrument Voigt params come from the CSV itself
    gamma = df['gamma'].iloc[0]

    #plt.plot(wavelengths,skw)
    #plt.show()
    #raise SystemExit

    def instr_func(w):
        v = voigt_profile(w.to(u.m).value, sigma, gamma)
        return v / v.max()

    # ---- scattering geometry --------------------------------------------------
    probe_vec = np.array([1, 0, 0])                          # probe beam direction
    scatter_vec = np.array([np.cos(np.deg2rad(90)),           # collection direction;
                             np.sin(np.deg2rad(90)), 0])      # angle to probe_vec = scattering angle

    # Thomson scattering only measures velocity along k = k_scatter - k_probe
    # (this is exactly how plasmapy projects ion_vel/electron_vel internally),
    # so drift directions below are set along k_hat rather than an arbitrary axis.
    k_vec = scatter_vec - probe_vec
    k_hat = k_vec / np.linalg.norm(k_vec)

    ions = ['Ne-20 9+', 'H+', 'Xe-132 17+']
    n_ions = len(ions)

    settings = {
        "probe_wavelength": (526.5 * u.nm).to(u.m).value,     # probe laser wavelength
        "probe_vec":        probe_vec,
        "scatter_vec":      scatter_vec,
        "ions":             ions,                              # e.g. 'p+','D+','He-4 2+','C-12 4+','Ne-20 8+'...
        "ion_vdir":         np.tile(k_hat, (n_ions, 1)),        # one row per ion species
        "electron_vdir":    np.tile(k_hat, (1, 1)),             # one row per electron population
        "instr_func":       instr_func,
        #"notch": (np.array([526.25, 526.6]) * u.nm).to(u.m).value,  # optional: zero out a wavelength range
    }

    # ---- params (fit variables) ----------------------------------------------
    params = Parameters()
    params.add('n', value=5e17 * 1e6, min=1e23, max=1e27, vary=True)   # total electron density, m^-3

    # --- per-ion-species: T_i_<i>, ion_speed_<i> ---
    params.add('T_i_0', value=150, min=10, max=5000, vary=True)        # eV
    params.add('T_i_1', value=150, min=1, max=500, vary=True)        # eV
    params.add('T_i_2', expr='T_i_1')#value=100, min=0.5, max=500, vary=True)        # eV
    params.add('ion_speed_0', value=0, min=-5e5, max=5e5, vary=True)   # m/s along ion_vdir[0] (= k_hat)
    params.add('ion_speed_1', value=0, min=-5e5, max=5e5, vary=True)   # m/s along ion_vdir[1] (= k_hat)
    params.add('ion_speed_2', value=0, min=-5e5, max=5e5, vary=True)   # m/s along ion_vdir[1] (= k_hat)
    # params.add('ion_mu_0', value=20, vary=False)   # optional override: ion mass / proton mass
    # params.add('ion_z_0',  value=8,  vary=False)   # optional override: ion charge number
    # (ion_mu_<i>/ion_z_<i> default to whatever `ions` implies; only set to override)

    params.add('ifract_0', value=0.4, min=0, max=0.6, vary=True)
    params.add('ifract_1', expr='(1 - ifract_0) * 1/(1 + 0.0065)')
    params.add('ifract_2', expr='(1 - ifract_0) * 0.0065/(1 + 0.0065)')
    '''
    ifract_guess = [0.5, 0.45, 0.05]   # <- edit initial guesses, must sum to 1, length = n_ions
    remaining = 1.0
    for i in range(n_ions - 1):
        params.add(f'sfract_{i}', value=ifract_guess[i] / remaining, min=0, max=1, vary=True)
        remaining -= ifract_guess[i]
    for i in range(n_ions - 1):
        expr = f'sfract_{i}' if i == 0 else f'sfract_{i} * ' + ' * '.join(f'(1 - sfract_{j})' for j in range(i))
        params.add(f'ifract_{i}', expr=expr)
    params.add(f'ifract_{n_ions - 1}', expr='1 - ' + ' - '.join(f'ifract_{i}' for i in range(n_ions - 1)))
    '''
    # --- electrons: T_e_<e>, electron_speed_<e> ---
    params.add('T_e_0', value=50, min=0.5, max=800, vary=True)             # eV
    params.add('electron_speed_0', value=0, min=-5e5, max=5e5, vary=True)   # m/s along electron_vdir[0] (= k_hat)
    # params.add('efract_0', value=1.0, vary=False)   # only needed with >1 electron population
    # params.add('T_e_1', value=200, min=0.5, max=5000, vary=True)          # (same sfract trick applies to efract)

    # params.add('background', value=0.0, min=0, max=1, vary=True)   # optional constant offset, fraction of max

    # ---- fit (retry until R² > 0.97) -----------------------------------------
    r2 = -1
    tries=0

    while r2 < 0.98 and tries<4:

        model = thomson.spectral_density_model(
            wavelengths.to(u.m).value,
            settings,
            params
        )

        result = model.fit(
            skw,
            params=copy.deepcopy(params),
            wavelengths=wavelengths.to(u.m).value,
            method="differential_evolution"
        )

        r2 = 1 - result.residual.var() / np.var(skw)
        tries+=1
        print(r2)
    if r2 <0.98:
        continue

    # ---- plot -------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(wavelengths.value, skw, label="Data")
    ax.plot(wavelengths.value, result.best_fit, label="Best fit")
    ax.axvline((settings["probe_wavelength"] * u.m).to(u.nm).value, color="red")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Skw")
    ax.legend()

    # ---- results -----------------------------------------------------------------
    def _fmt(val, err, unit):
        return f"{val:.3g} ± {err:.3g} {unit}" if err is not None else f"{val:.3g} {unit}"

    def _fmt2(val, err, unit):
        return f"{val:.3g} ± {err:.3g} {unit}" if err is not None else f"{val:.3f} {unit}"

    p = result.params
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
    lines.append(f"red. χ²     = {result.redchi:.4g}")
    lines.append(f"R²          = {r2:.4f}")

    print("\n".join(lines))

    # ---- save only good fits ---------------------------------------------------
    PLOT_DIR = f"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts-out/{shot}"
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
    plt.plot(wavelengths.value, result.best_fit, 'k', lw=2, label='Total fit')

    best_params = result.params.copy()
    Z = [9, 1, 17]

    for i in range(n_ions):
        p = best_params.copy()

        for j in range(n_ions):
            p[f'ifract_{j}'].set(value=(1.0 if j == i else 0.0))

        spec = model.eval(params=p, wavelengths=wavelengths.to(u.m).value)

        weights = np.array([
            result.params[f'ifract_{j}'].value * Z[j]
            for j in range(n_ions)
        ])

        weights /= weights.sum()
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

    df_out = pd.DataFrame(out)

    OUTPUT_PATH = f"{PLOT_DIR}/fiber{fiber}_{num}.csv"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved results to {OUTPUT_PATH}")
