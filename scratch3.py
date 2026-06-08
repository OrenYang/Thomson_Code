# Thomson Scattering Analysis - Simple Version
# This code loads a plasma spectrum, calibrates it using neon lamp lines,
# and tries to fit an ion acoustic wave (IAW) model to the data.

import numpy as np
import matplotlib.pyplot as plt
import sif_parser

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
import astropy.units as u
from plasmapy.diagnostics import thomson
from lmfit import Parameters, Minimizer

# Known neon wavelengths in nanometers (used for calibration)
NEON_WAVELENGTHS_NM = np.array([522.23519, 523.40271, 527.40393, 528.00853, 529.81891])


# -------------------------------------------------------
# STEP 1: Load a .sif file from the Andor camera
# -------------------------------------------------------
def load_sif(filepath):
    """Loads a .sif file and returns the 2D image and info."""
    data, info = sif_parser.np_open(filepath)
    # Make sure we just get a 2D frame (rows x columns)
    frame = data[0] if data.ndim == 3 else np.squeeze(data)
    return frame, info


# -------------------------------------------------------
# STEP 2: Find the bright horizontal "bins" (rows) in the image
# -------------------------------------------------------
def find_bins(frame):
    """
    Looks at the row-by-row brightness to find where the signal is.
    Returns a list of (top_row, bottom_row, center_row) for each bin.
    """
    # Sum all columns together so we get a 1D brightness profile
    row_brightness = gaussian_filter1d(frame.sum(axis=1), sigma=2)

    # Find the bright peaks in that profile
    peaks, _ = find_peaks(row_brightness, distance=10, height=row_brightness.max() * 0.1)

    bins = []
    for peak_row in peaks:
        # Walk up from the peak until brightness stops increasing
        top = peak_row
        while top > 0 and row_brightness[top - 1] <= row_brightness[top]:
            top -= 1

        # Walk down from the peak until brightness stops increasing
        bottom = peak_row
        while bottom < len(row_brightness) - 1 and row_brightness[bottom + 1] <= row_brightness[bottom]:
            bottom += 1

        bins.append((top, bottom, peak_row))

    return bins


# -------------------------------------------------------
# STEP 3: Extract a 1D spectrum (lineout) from one bin
# -------------------------------------------------------
def get_lineout(frame, top_row, bottom_row):
    """
    Sums the rows in a bin to get a 1D spectrum.
    Also subtracts a simple background (median of the edges).
    """
    lineout = frame[top_row:bottom_row + 1, :].sum(axis=0).astype(float)

    # Simple background: average the first and last 50 pixels
    background = np.median(np.concatenate([lineout[:50], lineout[-50:]]))
    return lineout - background


# -------------------------------------------------------
# STEP 4: Fit a Voigt peak shape to a neon emission line
# -------------------------------------------------------
def fit_one_voigt_peak(lineout, peak_pixel, window=30):
    """
    Fits a Voigt profile (a mix of Gaussian and Lorentzian) to one peak.
    Returns sigma (Gaussian width) and gamma (Lorentzian width), or None if it fails.
    """
    num_cols = len(lineout)
    left = max(0, peak_pixel - window)
    right = min(num_cols, peak_pixel + window)

    x = np.arange(left, right, dtype=float)
    y = lineout[left:right]

    try:
        # voigt_profile(x - center, sigma, gamma)
        def voigt_func(x, amplitude, center, sigma, gamma, baseline):
            return amplitude * voigt_profile(x - center, sigma, gamma) + baseline

        initial_guess = [y.max(), float(peak_pixel), 2.0, 2.0, 0.0]
        bounds_low = [0, left, 0.1, 0.1, -np.inf]
        bounds_high = [np.inf, right, 20, 20, np.inf]

        popt, _ = curve_fit(voigt_func, x, y, p0=initial_guess, bounds=(bounds_low, bounds_high))
        sigma = popt[2]
        gamma = popt[3]
        return sigma, gamma, popt, left, right

    except RuntimeError:
        print(f"  Warning: Voigt fit failed at pixel {peak_pixel}")
        return None, None, None, left, right


# -------------------------------------------------------
# STEP 5: Calibrate -- turn pixel numbers into wavelengths
# -------------------------------------------------------
def calibrate(cal_filepath, plot=True):
    """
    Loads a neon calibration file and figures out
    which wavelength corresponds to each pixel column.
    Returns a dictionary: bin_index -> array of wavelengths (nm)
    Also returns instrument function info (how wide the laser line is).
    """
    frame, _ = load_sif(cal_filepath)
    num_cols = frame.shape[1]
    bins = find_bins(frame)

    wavelength_arrays = {}       # bin -> array of wavelength values (nm)
    instrument_functions = {}    # bin -> widths of instrument response

    print(f"Calibration: found {len(bins)} bins")

    for bin_idx, (top, bottom, _) in enumerate(bins):
        lineout = get_lineout(frame, top, bottom)

        # Find peaks in this lineout
        peaks, _ = find_peaks(lineout, distance=5, height=lineout.max() * 0.05)

        if len(peaks) < 5:
            print(f"  Bin {bin_idx}: not enough peaks, skipping")
            continue

        # Use the 5 brightest peaks, in left-to-right order
        top5 = np.sort(peaks[np.argsort(lineout[peaks])[-5:]])

        # Fit a polynomial: pixel position -> wavelength (nm)
        poly_coeffs = np.polyfit(top5, NEON_WAVELENGTHS_NM, deg=2)
        wl_axis = np.polyval(poly_coeffs, np.arange(num_cols))
        wavelength_arrays[bin_idx] = wl_axis

        # Check how well the fit worked (residuals = errors in nm)
        residuals = np.polyval(poly_coeffs, top5) - NEON_WAVELENGTHS_NM
        print(f"  Bin {bin_idx}: {wl_axis[0]:.2f} to {wl_axis[-1]:.2f} nm")
        print(f"    Calibration residuals (nm): {residuals.round(4)}")

        # Fit Voigt peaks to measure the instrument function width
        sigmas, gammas = [], []
        for px in top5:
            sigma, gamma, popt, lo, hi = fit_one_voigt_peak(lineout, px)
            if sigma is not None:
                sigmas.append(sigma)
                gammas.append(gamma)

        if sigmas:
            # Convert pixel widths to nm using the local dispersion
            dispersion = abs(wl_axis[num_cols // 2 + 1] - wl_axis[num_cols // 2])  # nm per pixel
            sigma_nm = np.mean(sigmas) * dispersion
            gamma_nm = np.mean(gammas) * dispersion
            # FWHM formula for a Voigt profile
            fwhm_nm = 0.5346 * 2 * gamma_nm + np.sqrt(0.2166 * (2 * gamma_nm)**2 + (2.355 * sigma_nm)**2)
            instrument_functions[bin_idx] = {
                'sigma_nm': sigma_nm,
                'gamma_nm': gamma_nm,
                'fwhm_nm': fwhm_nm
            }
            print(f"    Instrument FWHM: {fwhm_nm:.4f} nm")

        if plot:
            plt.figure(figsize=(10, 3))
            plt.plot(wl_axis, lineout)
            for wl in NEON_WAVELENGTHS_NM:
                plt.axvline(wl, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.title(f'Bin {bin_idx} calibration')
            plt.tight_layout()
            plt.show()

    return wavelength_arrays, instrument_functions


# -------------------------------------------------------
# STEP 6: Fit the Thomson IAW spectrum for one bin
# -------------------------------------------------------
def fit_iaw(
    wavelengths_nm,      # wavelength axis for this bin (nm)
    lineout,             # measured spectrum (counts)
    probe_wavelength_nm=526.5,
    scattering_angle_deg=90,
    ions=None,
    n_e=1e18,            # electron density in cm^-3
    T_e=200,             # electron temperature in eV
    T_i=None,            # ion temperatures in eV (one per ion species)
    ifract=None,         # ion fractions (must add up to 1)
    ion_speeds=None,     # ion drift speeds in km/s
    plot=True,
    bin_label=''
):
    """
    Fits the ion acoustic wave (IAW) feature in a Thomson scattering spectrum.
    Adjusts T_i (ion temperature) and ion fractions to match the data.
    """
    if ions is None:
        ions = ["H+"]
    n_ions = len(ions)

    # Fill in defaults if not provided
    if T_i is None:        T_i = [10.0] * n_ions
    if ifract is None:     ifract = [1.0 / n_ions] * n_ions
    if ion_speeds is None: ion_speeds = [0.0] * n_ions

    # Scattering geometry: probe beam direction and detector direction
    probe_vec = np.array([1, 0, 0])
    angle_rad = np.deg2rad(scattering_angle_deg)
    scatter_vec = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])

    # Clean the data: remove non-finite values, normalize to max = 1
    valid = np.isfinite(lineout)
    wl = wavelengths_nm[valid]
    intensity = lineout[valid]
    intensity = intensity / np.nanmax(intensity)

    wl_qty = wl * u.nm
    probe_wl = probe_wavelength_nm * u.nm

    # This function calls PlasmaPy to compute the theoretical spectrum
    def compute_spectrum(T_i_vals, ifract_vals):
        _, skw = thomson.spectral_density(
            wl_qty, probe_wl,
            n_e * u.cm**-3,
            T_e=T_e * u.eV,
            T_i=np.array(T_i_vals) * u.eV,
            ions=ions,
            ifract=ifract_vals,
            electron_vel=np.array([[0, 0, 0]]) * u.km / u.s,
            ion_vel=np.array([[v, 0, 0] for v in ion_speeds]) * u.km / u.s,
            probe_vec=probe_vec,
            scatter_vec=scatter_vec,
        )
        skw = skw.value
        return skw / np.nanmax(skw)   # normalize

    # This is the function lmfit will minimize (returns data - model)
    def residual(params):
        T_i_vals = [params[f"T_i_{k}"].value for k in range(n_ions)]

        # Reconstruct all fractions; last one = 1 - sum of the rest
        raw_fracs = [params[f"ifract_{k}"].value for k in range(n_ions - 1)]
        last_frac = 1.0 - sum(raw_fracs)
        ifract_vals = raw_fracs + [last_frac]

        # Reject unphysical fractions
        if last_frac < 0.01 or last_frac > 0.98:
            return np.full(len(intensity), 1e6)

        try:
            skw = compute_spectrum(T_i_vals, ifract_vals)
            if not np.all(np.isfinite(skw)):
                return np.full(len(intensity), 1e6)
            return intensity - skw
        except Exception:
            return np.full(len(intensity), 1e6)

    # Set up fitting parameters with initial guesses and allowed ranges
    params = Parameters()
    for k in range(n_ions):
        params.add(f"T_i_{k}", value=T_i[k], vary=True, min=1, max=500)
    for k in range(n_ions - 1):
        params.add(f"ifract_{k}", value=ifract[k], vary=True, min=0.01, max=0.98)

    # Run the fit
    minimizer = Minimizer(residual, params)
    result = minimizer.minimize(method="leastsq")

    # Pull out the best-fit values
    best_T_i = [result.params[f"T_i_{k}"].value for k in range(n_ions)]
    raw_fracs = [result.params[f"ifract_{k}"].value for k in range(n_ions - 1)]
    best_ifract = raw_fracs + [1.0 - sum(raw_fracs)]
    best_spectrum = compute_spectrum(best_T_i, best_ifract)

    # Print a summary
    print(f"\nBin {bin_label} IAW fit result:")
    for k, ion in enumerate(ions):
        print(f"  {ion}: T_i = {best_T_i[k]:.2f} eV,  fraction = {best_ifract[k]:.3f}")
    print(f"  Reduced chi² = {result.redchi:.4f}  ({result.nfev} iterations)")

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(wl, intensity, label='Data')
        plt.plot(wl, best_spectrum, color='red', label='Best fit')
        plt.axvline(probe_wavelength_nm, color='gray', linestyle='--', label='Probe')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized intensity')
        plt.title(f'Bin {bin_label} — IAW fit')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result, best_T_i, best_ifract


# -------------------------------------------------------
# MAIN: Run everything
# -------------------------------------------------------
def main():
    # -- Load the Thomson scattering data --
    shot_file = 'data/07230_TS_shotn.sif'
    frame, info = load_sif(shot_file)
    print(f"Loaded frame: {frame.shape[0]} rows x {frame.shape[1]} columns")

    # -- Find bins and extract lineouts --
    bins = find_bins(frame)
    print(f"\nFound {len(bins)} bins:")
    for i, (top, bottom, center) in enumerate(bins):
        print(f"  Bin {i}: rows {top}–{bottom} (center row {center})")

    lineouts = {}
    for i, (top, bottom, _) in enumerate(bins):
        lineouts[i] = get_lineout(frame, top, bottom)

    # -- Calibrate using a neon lamp file --
    cal_file = 'data/07230_TS_necaln.sif'
    wavelengths, instrument_funcs = calibrate(cal_file, plot=False)

    # -- Fit IAW for each bin --
    for bin_idx in range(min(10, len(bins))):
        if bin_idx not in wavelengths or bin_idx not in lineouts:
            print(f"Bin {bin_idx}: missing calibration or data, skipping")
            continue

        fit_iaw(
            wavelengths_nm=wavelengths[bin_idx],
            lineout=lineouts[bin_idx],
            probe_wavelength_nm=526.5,
            scattering_angle_deg=90,
            ions=["H+", "Xe 8+", "Ne 8+"],
            n_e=1e18,
            T_e=200,
            T_i=[100, 100, 150],
            ifract=[0.6, 0.05, 0.35],
            ion_speeds=[0, 0, 0],
            bin_label=str(bin_idx)
        )


if __name__ == '__main__':
    main()
def fit_iaw(self, bin_index=0, probe_wavelength=526.5, scattering_angle_deg=90,
            iaw_window_nm=2.0, ne_guess=1e18, T_e_guess=40, T_i_guess=20,
            v_guess=0.0):
    """Requires calibrate() to have been called first."""
    bin = self.bins[bin_index]
    mask = np.abs(bin.wavelength - probe_wavelength) <= iaw_window_nm
    wl_fit = bin.wavelength[mask]
    I_fit = bin.spectrum[mask] / bin.spectrum[mask].max()

    angle = np.deg2rad(scattering_angle_deg)
    scatter_vec = np.array([np.cos(angle), np.sin(angle), 0])

    settings = {
        "probe_wavelength": (probe_wavelength * u.nm).to(u.m).value,
        "probe_vec": np.array([1, 0, 0]),
        "scatter_vec": scatter_vec,
        "ions": ["p+"],
        "ion_vdir": np.array([scatter_vec / np.linalg.norm(scatter_vec)]),  # shape (1, 3)
    }

    cal_bin = self.cal.bins[bin_index]
    nm_per_px = np.mean(np.diff(cal_bin.wavelength))
    m_per_px = nm_per_px * 1e-9
    s_m = cal_bin.s * m_per_px  # sigma in metres
    g_m = cal_bin.g * m_per_px  # gamma in metres

    settings["instr_func"] = lambda wl: voigt_profile(
        np.asarray(wl), s_m, g_m  # wl already in metres, s/g now in metres
    )

    params = Parameters()
    params.add("n",         value=(ne_guess * u.cm**-3).to(u.m**-3).value, vary=False)
    params.add("T_e_0",     value=T_e_guess, vary=True)
    params.add("T_i_0",     value=T_i_guess, vary=True, min=1, max=500)
    params.add("efract_0",  value=1.0, vary=False)
    params.add("ifract_0",  value=1.0, vary=False)
    params.add("ion_speed_0", value=v_guess, vary=True, min=-5e6, max=5e6)  # m/s

    wl_m = (wl_fit * u.nm).to(u.m).value
    iaw_model = thomson.spectral_density_model(wl_m, settings, params)
    result = iaw_model.fit(I_fit, params, wavelengths=wl_m)

    print(result.fit_report())

    v_fit  = result.params["ion_speed_0"].value
    v_err  = result.params["ion_speed_0"].stderr
    Ti_fit = result.params["T_i_0"].value
    Ti_err = result.params["T_i_0"].stderr

    fig, ax = plt.subplots()
    ax.plot(wl_fit, I_fit, 'k.', ms=3, label='data')
    ax.plot(wl_fit, result.best_fit, 'r-', lw=2, label='fit')
    ax.set(xlabel=r"$\lambda$ (nm)", ylabel="Intensity (norm.)",
           title=fr"$T_i$={Ti_fit:.1f}±{Ti_err:.1f} eV,  $v$={v_fit/1e3:.0f}±{v_err/1e3:.0f} km/s")
    ax.legend()
    plt.show()

    return result




    def fit_iaw(self, probe_wl=526.5, iaw_window_nm=2, probe_vec=np.array([1,0,0]),
                    angle=90, ions=["H+"], electron_vdir=None, ion_vdir=None,
                    instr_func=None, notch=None,
                    n=1e18, vary_n=True, T_e_0 = 100, vary_T_e = True, electron_speed=None, vary_electron_speed=True,
                    T_i_=[10], vary_T_i_=None, ifract_=[1], vary_ifract_=None, ion_speed_=[10], vary_ion_speed_=None):
        for bin in self.bins:
            ##### Initialize calibrated experimental spectrum ########
            mask = np.abs(bin.wavelength - probe_wl) <= iaw_window_nm
            nan_mask = ~np.isnan(bin.spectrum[mask]) & ~np.isnan(bin.wavelength[mask])
            wl = (bin.wavelength[mask][nan_mask] * u.nm).to(u.m)
            iaw_skw = bin.spectrum[mask][nan_mask]
            iaw_skw = iaw_skw / iaw_skw.max()

            ######## Create Settings dict for the model input ########
            probe_wavelength = (probe_wl * u.nm).to(u.m).value # convert from nm to m
            scattering_angle = np.deg2rad(angle)
            scattering_vec = np.array([np.cos(scattering_angle),np.sin(scattering_angle),0])
            if ion_vdir is None:
                ion_vdir = []
                for i in range(len(ions)):
                    ion_vdir.append([scattering_vec/np.linalg.norm(scattering_vec)])
            if electron_vdir is None:
                electron_vdir = []
                for i in range(len(ions)):
                    electron_vdir.append([scattering_vec/np.linalg.norm(scattering_vec)])
            s = 0
            g = 0
            for b in self.cal.bins:
                if b.fiber == bin.fiber:
                    s = b.s
                    g = b.g
            settings = {k: v for k, v in {
                "probe_wavelength": probe_wavelength,
                "probe_vec": probe_vec,
                "scatter_vec": scattering_vec,
                "ions": ions,
                "electron_vdir": electron_vdir,
                "ion_vdir": ion_vdir,
                "instr_func": (lambda wl_fit: voigt_profile(np.asarray(wl_fit),s, g)),
                "notch": notch,
            }.items() if v is not None}

            ###### Create Parameters object for the model input #######
            params = Parameters()
            params.add("n", value = n, vary = vary_n)
            params.add("T_e_0", value = T_e_0, vary = vary_T_e)
            #params.add("efract_0")
            if electron_speed is not None:
                params.add("electron_speed_0", value = electron_speed, vary = vary_electron_speed)
            for i in range(len(ions)):
                if vary_T_i_ is None:
                    vary_Ti = True
                else:
                    vary_Ti = vary_T_i_[i]
                if vary_ifract_ is None:
                    vary_ifract = True
                else:
                    vary_ifract = vary_ifract_[i]
                if vary_ion_speed_ is None:
                    vary_ion_speed = True
                else:
                    vary_ion_speed = vary_ion_speed_[i]
                params.add(f"T_i_{i}", value = T_i_[i], vary = vary_Ti)
                params.add(f"ifract_{i}", value = ifract_[i], vary = vary_ifract)
                params.add(f"ion_speed_{i}", value = ion_speed_[i], vary = vary_ion_speed)
                #params.add(f"ion_mu_{i}")
                #params.add(f"ion_z_{i}")
            #params.add("background")

            ################# Create the model and fit #################
            iaw_model = thomson.spectral_density_model(wl.value,settings,params)

            iaw_results = iaw_model.fit(
                iaw_skw,
                params=params,
                wavelengths=wl.value,
                method="leastsq",
            )

            print(f"Number of fit iterations:{iaw_results.nfev:.1f}")
            print(f"Reduced Chisquared:{iaw_results.redchi:.4f}")

            # Extract the best fit curve
            best_fit_skw = iaw_result.best_fit

            # Plot
            fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Skw")
            ax.axvline(x=probe_wavelength.value, color="red", label="Probe wavelength")

            ax.set_xlim(probe_wl - iaw_window_nm, probe_wl + iaw_window_nm)

            wl_nm = wl.to(u.nm)  # convert back for a readable x-axis
            ax.plot(wl_nm.value, iaw_skw, label="Data")
            ax.plot(wl_nm.value, best_fit_skw, label="Best-fit")
            ax.legend(loc="upper right");



    def fit_iaw(self, probe_wavelength=526.5, iaw_window_nm=2.0,
                ne=1e18, T_e=100.0, T_i=20.0, v_flow=-100.0,
                ion_species=('H+',), ion_fractions=None,
                fit_composition=True,
                shared_flow=True,    # single bulk v for all species
                shared_T_i=False,    # single T_i for all species
                scattering_angle_deg=90, plot=True):
        from plasmapy.diagnostics.thomson import spectral_density_model

        n_species = len(ion_species)

        if ion_fractions is None:
            ion_fractions = np.ones(n_species) / n_species
        else:
            ion_fractions = np.asarray(ion_fractions, dtype=float)
            ion_fractions /= ion_fractions.sum()

        T_i_arr    = np.broadcast_to(np.atleast_1d(T_i),    (n_species,)).copy()
        v_flow_arr = np.broadcast_to(np.atleast_1d(v_flow), (n_species,)).copy()

        angle       = np.deg2rad(scattering_angle_deg)
        probe_vec   = np.array([1, 0, 0])
        scatter_vec = np.array([np.cos(angle), np.sin(angle), 0])
        k_vec       = scatter_vec - probe_vec
        k_hat       = k_vec / np.linalg.norm(k_vec)

        for b in self.bins:
            if not hasattr(b, 'wavelength'):
                print(f"Bin {b.fiber}: not calibrated, skipping.")
                continue

            mask    = np.abs(b.wavelength - probe_wavelength) <= iaw_window_nm
            wl_m    = b.wavelength[mask] * 1e-9
            sp_data = b.spectrum[mask].astype(float)

            if sp_data.max() <= 0:
                print(f"Bin {b.fiber}: non-positive spectrum in window, skipping.")
                continue

            sp_norm = sp_data / sp_data.max()

            settings = {
                'probe_wavelength' : probe_wavelength * 1e-9,
                'probe_vec'        : probe_vec,
                'scatter_vec'      : scatter_vec,
                'ions'             : list(ion_species),
                'ion_vdir'         : np.tile(k_hat, (n_species, 1)),
                'electron_vdir'    : k_hat.reshape(1, 3),
            }

            params = Parameters()
            params.add('n',               value=ne * 1e6,      min=1e19,  max=1e27,   vary=True)
            params.add('T_e_0',           value=T_e,           min=1.0,   max=2000.0, vary=True)
            params.add('background',      value=0.0,           min=-0.2,  max=0.2,    vary=True)

            # ── Ion fractions: stick-breaking reparameterisation ─────────────
            # Each z_i = ifract_i / (remaining budget) lives in (0,1) freely.
            # All derived ifract_{i} are guaranteed positive at every iteration.
            if fit_composition and n_species > 1:
                z_init = _stick_breaking_init(ion_fractions)
                remaining_expr = '1.0'
                for i in range(n_species - 1):
                    params.add(f'z_{i}',      value=np.clip(z_init[i], 1e-4, 1-1e-4),
                                              min=1e-4, max=1-1e-4, vary=True)
                    params.add(f'ifract_{i}', expr=f'z_{i} * ({remaining_expr})')
                    remaining_expr = f'({remaining_expr}) * (1 - z_{i})'
                params.add(f'ifract_{n_species-1}', expr=remaining_expr)
            else:
                for i in range(n_species):
                    params.add(f'ifract_{i}', value=ion_fractions[i], vary=False)

            # ── Shared or per-species flow ────────────────────────────────────
            if shared_flow:
                # One free bulk velocity; all ion_speed_{i} point to it
                params.add('v_flow',          value=v_flow_arr[0], min=-5e5, max=5e5, vary=True)
                params.add('electron_speed_0',expr='v_flow')
                for i in range(n_species):
                    params.add(f'ion_speed_{i}', expr='v_flow')
            else:
                params.add('electron_speed_0', value=v_flow_arr[0], min=-5e5, max=5e5, vary=True)
                for i in range(n_species):
                    params.add(f'ion_speed_{i}', value=v_flow_arr[i], min=-5e5, max=5e5, vary=True)

            # ── Shared or per-species T_i ─────────────────────────────────────
            if shared_T_i:
                # One free ion temperature; all T_i_{i} point to it
                params.add('T_i_shared', value=T_i_arr[0], min=0.5, max=500.0, vary=True)
                for i in range(n_species):
                    params.add(f'T_i_{i}', expr='T_i_shared')
            else:
                for i in range(n_species):
                    params.add(f'T_i_{i}', value=T_i_arr[i], min=0.5, max=500.0, vary=True)

            model  = spectral_density_model(wl_m * u.m, settings, params)
            result = model.fit(sp_norm, params=params, wavelengths=wl_m,
                               method='least_squares')

            # ── Store results ─────────────────────────────────────────────────
            b.T_e     = result.params['T_e_0'].value
            b.T_e_err = result.params['T_e_0'].stderr
            b.ne      = result.params['n'].value * 1e-6
            b.ne_err  = result.params['n'].stderr * 1e-6 if result.params['n'].stderr else None
            b.T_i     = [result.params[f'T_i_{i}'].value          for i in range(n_species)]
            b.T_i_err = [result.params[f'T_i_{i}'].stderr         for i in range(n_species)]
            b.v_flow  = result.params['v_flow'].value if shared_flow else \
                        [result.params[f'ion_speed_{i}'].value     for i in range(n_species)]
            b.v_flow_err = result.params['v_flow'].stderr if shared_flow else \
                           [result.params[f'ion_speed_{i}'].stderr for i in range(n_species)]
            b.ifract  = [result.params[f'ifract_{i}'].value       for i in range(n_species)]
            b.iaw_fit_result = result

            # ── Print ─────────────────────────────────────────────────────────
            def _fmt(val, err, unit):
                return (f"{val:.2f} ± {err:.2f} {unit}"
                        if err is not None else f"{val:.2f} {unit}")

            print(f"Bin {b.fiber}:")
            print(f"  T_e = {_fmt(b.T_e, b.T_e_err, 'eV')}  |  "
                  f"ne = {_fmt(b.ne/1e18, b.ne_err/1e18 if b.ne_err else None, '×10¹⁸ cm⁻³')}  |  "
                  f"redchi = {result.redchi:.4f}")
            if shared_flow:
                print(f"  v  = {_fmt(b.v_flow/1e3, b.v_flow_err/1e3 if b.v_flow_err else None, 'km/s')}  (shared)")
            for i, sp in enumerate(ion_species):
                frac_str = f"  ifract={b.ifract[i]:.3f}" if fit_composition else ""
                v_str = "" if shared_flow else \
                    f"  |  v = {_fmt(b.v_flow[i]/1e3, b.v_flow_err[i]/1e3 if b.v_flow_err[i] else None, 'km/s')}"
                print(f"  [{sp}]{frac_str}  T_i = {_fmt(b.T_i[i], b.T_i_err[i], 'eV')}{v_str}")

            # ── Plot ──────────────────────────────────────────────────────────
            if plot:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(b.wavelength[mask], sp_norm,         'k.', ms=3, label='Data (norm.)')
                v_val = b.v_flow if shared_flow else b.v_flow[0]
                v_err = b.v_flow_err if shared_flow else b.v_flow_err[0]
                label_lines = [
                    f"$T_e$={b.T_e:.1f} eV,  $n_e$={b.ne/1e18:.2f}×10¹⁸ cm⁻³",
                    f"$v$={v_val/1e3:.1f} km/s" + (" (shared)" if shared_flow else ""),
                ]
                for i, sp in enumerate(ion_species):
                    frac_str = f" ({b.ifract[i]:.2f})" if fit_composition else ""
                    v_str = "" if shared_flow else f", $v$={b.v_flow[i]/1e3:.1f} km/s"
                    label_lines.append(f"[{sp}]{frac_str} $T_i$={b.T_i[i]:.1f} eV{v_str}")
                ax.plot(b.wavelength[mask], result.best_fit, 'r-', lw=1.5,
                        label='\n'.join(label_lines))
                ax.axhline(0,                color='grey', ls=':',  lw=0.8)
                ax.axvline(probe_wavelength, color='grey', ls='--', lw=0.8)
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Normalised intensity')
                ax.set_title(f'IAW fit — Bin (fiber {b.fiber})')
                ax.legend(fontsize=8)
                plt.tight_layout()
                plt.show()
def _stick_breaking_init(fractions):
    """Convert a normalised fraction array to stick-breaking z values in (0,1)."""
    fractions = np.asarray(fractions, dtype=float)
    fractions /= fractions.sum()
    z = np.zeros(len(fractions) - 1)
    remaining = 1.0
    for i in range(len(fractions) - 1):
        z[i] = fractions[i] / remaining
        remaining -= fractions[i]
    return z


def fit_iaw(self, probe_wavelength=526.5, iaw_window_nm=2.0,
            ne=1e18, T_e=100.0, T_i=20.0, v_flow=-100.0,
            ion_species='H+', scattering_angle_deg=90, plot=True):
    from plasmapy.diagnostics.thomson import spectral_density_model

    angle       = np.deg2rad(scattering_angle_deg)
    probe_vec   = np.array([1, 0, 0])
    scatter_vec = np.array([np.cos(angle), np.sin(angle), 0])
    k_vec = scatter_vec - probe_vec
    k_hat = k_vec / np.linalg.norm(k_vec)

    for b in self.bins:
        if not hasattr(b, 'wavelength'):
            print(f"Bin {b.fiber}: not calibrated, skipping.")
            continue

        mask    = np.abs(b.wavelength - probe_wavelength) <= iaw_window_nm
        wl_m    = b.wavelength[mask] * 1e-9      # plain float64 array, metres
        sp_data = b.spectrum[mask].astype(float)

        if sp_data.max() <= 0:
            print(f"Bin {b.fiber}: non-positive spectrum in window, skipping.")
            continue

        sp_norm = sp_data / sp_data.max()

        # ── settings: ALL values must be plain floats/arrays, no Quantities ──
        settings = {
            'probe_wavelength' : probe_wavelength * 1e-9,   # bare float, metres
            'probe_vec'        : probe_vec,
            'scatter_vec'      : scatter_vec,
            'ions'             : [ion_species],
            'ion_vdir'         : k_hat.reshape(1, 3),
            'electron_vdir'    : k_hat.reshape(1, 3),
        }

        params = Parameters()
        params.add('n',               value=ne * 1e6, min=1e19,  max=1e27,   vary=True)
        params.add('T_e_0',           value=T_e,      min=1.0,   max=2000.0, vary=True)
        params.add('T_i_0',           value=T_i,      min=0.5,   max=500.0,  vary=True)
        params.add('ion_speed_0',     value=v_flow,   min=-5e5,   max=5e5,    vary=True)
        params.add('electron_speed_0',value=v_flow,   min=-5e5,   max=5e5,    vary=True)
        params.add('background',      value=0.0,      min=-0.2,  max=0.2,    vary=True)

        model  = spectral_density_model(wl_m * u.m, settings, params)
        result = model.fit(sp_norm, params=params, wavelengths=wl_m,
                           method='least_squares')

        b.T_i    = result.params['T_i_0'].value
        b.T_e    = result.params['T_e_0'].value
        b.ne     = result.params['n'].value * 1e-6
        b.v_flow = result.params['ion_speed_0'].value
        b.iaw_fit_result = result

        b.T_i_err    = result.params['T_i_0'].stderr
        b.T_e_err    = result.params['T_e_0'].stderr
        b.ne_err     = (result.params['n'].stderr * 1e-6
                        if result.params['n'].stderr else None)
        b.v_flow_err = result.params['ion_speed_0'].stderr

        def _fmt(val, err, unit):
            return (f"{val:.2f} ± {err:.2f} {unit}"
                    if err is not None else f"{val:.2f} {unit}")

        print(
            f"Bin {b.fiber}: "
            f"T_i = {_fmt(b.T_i, b.T_i_err, 'eV')}  |  "
            f"T_e = {_fmt(b.T_e, b.T_e_err, 'eV')}  |  "
            f"ne = {_fmt(b.ne/1e18, b.ne_err/1e18 if b.ne_err else None, '×10¹⁸ cm⁻³')}  |  "
            f"v = {_fmt(b.v_flow/1e3, b.v_flow_err/1e3 if b.v_flow_err else None, 'km/s')}  |  "
            f"redchi = {result.redchi:.4f}"
        )

        if plot:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(b.wavelength[mask], sp_norm,         'k.', ms=3, label='Data (norm.)')
            ax.plot(b.wavelength[mask], result.best_fit, 'r-', lw=1.5,
                    label=(f"$T_i$={b.T_i:.1f} eV, $T_e$={b.T_e:.1f} eV\n"
                           f"$n_e$={b.ne/1e18:.2f}×10¹⁸ cm⁻³, "
                           f"$v$={b.v_flow/1e3:.1f} km/s"))
            ax.axhline(0,                color='grey', ls=':',  lw=0.8)
            ax.axvline(probe_wavelength, color='grey', ls='--', lw=0.8)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Normalised intensity')
            ax.set_title(f'IAW fit — Bin (fiber {b.fiber})')
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)
            plt.close()

def create_synthetic_spectrum(probe_wavelength=526.5, ne=1e18, T_e=100, T_i=20, scattering_angle_deg=90, iaw_window_nm=2.0):
    pw = probe_wavelength * u.nm
    wl = np.arange(probe_wavelength - iaw_window_nm * 2, probe_wavelength + iaw_window_nm * 2, 0.001) * u.nm
    angle = np.deg2rad(scattering_angle_deg)
    alpha, Skw = thomson.spectral_density(wl, pw, ne * u.cm**-3, T_e=T_e * u.eV, T_i=T_i * u.eV,
        probe_vec=np.array([1, 0, 0]), scatter_vec=np.array([np.cos(angle), np.sin(angle), 0]))
    mask = np.abs(wl.value - probe_wavelength) <= iaw_window_nm
    Skw = Skw.value
    fig, ax = plt.subplots()
    ax.plot(wl.value[mask], Skw[mask])
    ax.set(xlabel=r"$\lambda$ (nm)", ylabel=r"$S(k,\omega)$", ylim=(0, Skw[mask].max() * 1.1))
    plt.show()
