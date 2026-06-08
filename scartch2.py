import sif_parser
import matplotlib
matplotlib.use("TkAgg")
from tkinter import filedialog, Tk
from dataclasses import dataclass, field

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.special import voigt_profile
from scipy.optimize import curve_fit

from plasmapy.diagnostics import thomson

KNOWN_NEON_WL = np.array([522.23519, 523.40271, 527.40393, 528.00853, 529.81891]) * 1e-7  # cm

def main():
    test = Thomson('data/07230_TS_shotn.sif')
    #cal = load_calfile('data/07230_TS_necaln.sif', plot=False)
    #cal = recalibrate(cal, [12])
    test.create_lineouts(plot_bins=True)
    '''test.calibrate(cal)
    for i in range(10):
        test.fit_iaw(
            bin_idx=i,
            probe_wavelength_nm=526.5,
            scattering_angle_deg=90,
            ions=["H+", "Xe 8+", "Ne 8+"],
            n=1e18,
            T_e=200,
            T_i=[100, 100, 150],
            ifract=[0.6, 0.05, 0.35],
            ion_speeds=[0, 0, 0],
        )'''

@dataclass
class CalibrationResult:
    wavelength_arrays: dict        # bin_idx -> wavelength axis (nm)
    instrument_functions: dict     # bin_idx -> {sigma_nm, gamma_nm, fwhm_nm, dispersion_nm_per_px}
    frame: np.ndarray              # raw calibration frame
    bins: list                     # [(lo, hi, peak_row), ...]
    residuals: dict = field(default_factory=dict)  # bin_idx -> residual array

class Thomson:
    def __init__(self, sif=None):
        if sif is None: sif = open_sif("Select a .sif file")
        self.frame, self.info = load_frame(sif)
        self.H, self.W = self.frame.shape

    def create_lineouts(self, plot_bins=False, plot_lineouts=True):
        bins = find_bins(self.frame)
        print(f"Found {len(bins)} bins:")
        for i, (lo, hi, pk) in enumerate(bins):
            print(f"  Bin {i:2d}: rows {lo:4d}–{hi:4d} (center {pk})")

        if plot_bins:
            profile = gaussian_filter1d(self.frame.sum(axis=1), sigma=2)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(self.frame, cmap='inferno', origin='lower', aspect='auto')
            axes[0].set_title('Raw frame')
            axes[1].plot(profile, np.arange(self.H), color='white', lw=1.5)
            axes[1].set_facecolor('#111')
            for lo, hi, pk in bins:
                for ax in axes:
                    ax.axhline(lo, color='cyan', lw=0.8, ls='--')
                    ax.axhline(hi, color='cyan', lw=0.8, ls='--')
                    ax.axhline(pk, color='yellow', lw=0.8, ls=':')
            axes[0].set_xlabel('Column (px)'); axes[0].set_ylabel('Row (px)')
            axes[1].set_xlabel('Summed intensity'); axes[1].set_ylabel('Row (px)')
            axes[1].set_ylim(0, self.H)
            plt.tight_layout(); plt.show()

        self.lineouts = {i: get_lineout(self.frame, lo, hi) for i, (lo, hi, _) in enumerate(bins)}

        if plot_lineouts:
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, lineout in self.lineouts.items():
                ax.plot(lineout, label=f'Bin {i}')
            ax.set_xlabel('Column (px)'); ax.set_ylabel('Summed intensity')
            plt.tight_layout(); plt.legend(); plt.show()

    def calibrate(self, cal: CalibrationResult = None, calfile=None, plot=True):
        if cal is None:
            cal = load_calfile(f=calfile, plot=plot)
        self.wavelengths = cal.wavelength_arrays
        self.instrument_functions = cal.instrument_functions

    def fit_iaw(self, bin_idx, probe_wavelength_nm=526.5, scattering_angle_deg=90,
                ions=None, probe_vec=None, scatter_vec=None,
                n=2e17, T_e=10, T_i=None, ifract=None, ion_speeds=None,
                notch_range=None, plot=True):

        # --- setup ---
        if ions is None: ions = ["H+"]
        n_ions = len(ions)
        if T_i is None:        T_i = [10.0] * n_ions
        if ifract is None:     ifract = [1.0 / n_ions] * n_ions
        if ion_speeds is None: ion_speeds = [0.0] * n_ions

        if probe_vec is None: probe_vec = np.array([1, 0, 0])
        if scatter_vec is None:
            angle = np.deg2rad(scattering_angle_deg)
            scatter_vec = np.array([np.cos(angle), np.sin(angle), 0])

        wl_nm = self.wavelengths[bin_idx].copy()
        intensity = self.lineouts[bin_idx].copy()

        # --- notch / mask ---
        if notch_range is not None:
            mask = (wl_nm >= notch_range[0]) & (wl_nm <= notch_range[1])
            wl_nm = wl_nm[~mask]
            intensity = intensity[~mask]

        # --- clean ---
        valid = np.isfinite(intensity)
        intensity = intensity[valid]
        wl_nm = wl_nm[valid]
        intensity = intensity / np.nanmax(intensity)

        wl_qty = wl_nm * u.nm
        probe_wl = probe_wavelength_nm * u.nm

        def call_thomson(T_i_vals, ifract_vals):
            _, skw = thomson.spectral_density(
                wl_qty, probe_wl,
                n * u.cm**-3,
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
            skw = skw / np.nanmax(skw)
            return skw

        def residual(p):
            T_i_vals = [p[f"T_i_{k}"].value for k in range(n_ions)]
            raw_ifracts = [p[f"ifract_{k}"].value for k in range(n_ions - 1)]
            last = 1.0 - sum(raw_ifracts)
            ifract_vals = raw_ifracts + [last]
            if last < 0.01 or last > 0.98:
                return np.full(len(intensity), 1e6)
            try:
                skw = call_thomson(T_i_vals, ifract_vals)
                if skw.shape != intensity.shape or not np.all(np.isfinite(skw)):
                    return np.full(len(intensity), 1e6)
                return intensity - skw
            except Exception:
                return np.full(len(intensity), 1e6)

        # --- params ---
        params = Parameters()
        for k in range(n_ions):
            params.add(f"T_i_{k}", value=T_i[k], vary=True, min=1, max=500)
        for k in range(n_ions - 1):
            params.add(f"ifract_{k}", value=ifract[k], vary=True, min=0.01, max=0.98)

        from lmfit import Minimizer
        minimizer = Minimizer(residual, params)
        result = minimizer.minimize(method="leastsq")

        best_T_i = [result.params[f"T_i_{k}"].value for k in range(n_ions)]
        raw_ifracts = [result.params[f"ifract_{k}"].value for k in range(n_ions - 1)]
        best_ifract = raw_ifracts + [1.0 - sum(raw_ifracts)]
        best_skw = call_thomson(best_T_i, best_ifract)

        print(f"\nBin {bin_idx} IAW fit:")
        for k in range(n_ions):
            print(f"  T_i_{k}: {best_T_i[k]:.4g} eV")
        for k in range(n_ions):
            print(f"  ifract_{k}: {best_ifract[k]:.4g}")
        print(f"  Reduced chi²: {result.redchi:.4f}  ({result.nfev} iterations)")

        if plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(wl_nm, intensity, label='Data')
            ax.plot(wl_nm, best_skw, label='Best fit', color='red')
            ax.axvline(probe_wavelength_nm, color='gray', ls='--', lw=0.8, label='Probe')
            ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Normalised intensity')
            ax.set_title(f'Bin {bin_idx} — IAW fit')
            ax.legend(); plt.tight_layout(); plt.show()

        self.iaw_results = getattr(self, 'iaw_results', {})
        self.iaw_results[bin_idx] = result
        return result

    def plot(self, type='lineout', bins=None):
        if type == 'sif':
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(self.frame, cmap='inferno', origin='lower', aspect='auto')
            ax.set_xlabel('Column (px)'); ax.set_ylabel('Row (px)')
            ax.set_title('Raw frame')
            plt.tight_layout(); plt.show()
            return

        if not hasattr(self, 'lineouts') or not self.lineouts:
            print('No lineouts'); return

        all_items = list(self.lineouts.items())
        if bins is not None:
            all_items = [(i, lo) for i, lo in all_items if i in bins]

        bins_per_window = 5
        for start in range(0, len(all_items), bins_per_window):
            chunk = all_items[start:start + bins_per_window]
            fig, axes = plt.subplots(len(chunk), 1, sharex=False)
            if len(chunk) == 1: axes = [axes]
            for ax, (i, lineout) in zip(axes, chunk):
                wl = getattr(self, 'wavelengths', {}).get(i)
                ax.plot(wl if wl is not None else np.arange(len(lineout)), lineout, label=f'Bin {i}')
                ax.set_ylabel('Intensity')
                ax.legend(loc='upper right')
                if ax != axes[-1]:
                    ax.set_xticklabels([])
            axes[-1].set_xlabel('Wavelength (nm)' if wl is not None else 'Pixel')
            plt.tight_layout(); plt.show()

def open_sif(title):
    root = Tk(); root.withdraw(); root.update()
    f = filedialog.askopenfilename(title=title)
    root.destroy()
    if not f:
        raise ValueError("No file selected.")
    return f

def load_frame(path):
    data, info = sif_parser.np_open(path)
    return (data[0] if data.ndim == 3 else np.squeeze(data)), info

def find_bins(frame):
    profile = gaussian_filter1d(frame.sum(axis=1), sigma=2)
    peaks, _ = find_peaks(profile, distance=10, height=profile.max() * 0.1)
    bins = []
    for pk in peaks:
        lo = pk
        while lo > 0 and profile[lo-1] <= profile[lo]: lo -= 1
        hi = pk
        while hi < len(profile)-1 and profile[hi+1] <= profile[hi]: hi += 1
        bins.append((lo, hi, pk))
    return bins

def get_lineout(frame, lo, hi):
    lineout = frame[lo:hi+1, :].sum(axis=0).astype(float)
    bg = np.median(np.concatenate([lineout[:50], lineout[-50:]]))
    return lineout - bg

def fit_voigt(lineout, peak_px, window=30):
    W = len(lineout)
    sigmas, gammas = [], []
    fits = []
    for px in peak_px:
        lo, hi = max(0, px-window), min(W, px+window)
        x, y = np.arange(lo, hi, dtype=float), lineout[lo:hi]
        try:
            popt, _ = curve_fit(
                lambda x, a, c, s, g, b: a * voigt_profile(x-c, s, g) + b,
                x, y, p0=[y.max(), float(px), 2., 2., 0.],
                bounds=([0, lo, 0.1, 0.1, -np.inf], [np.inf, hi, 20, 20, np.inf])
            )
            sigmas.append(popt[2]); gammas.append(popt[3])
            fits.append((lo, hi, popt))
        except RuntimeError:
            fits.append((lo, hi, None))
    return sigmas, gammas, fits

def load_calfile(f=None, known_neon_wavelengths=KNOWN_NEON_WL, instrument_func=True, plot=True):
    if f is None: f = open_sif("Select a .sif calibration file")
    frame, _ = load_frame(f)
    H, W = frame.shape
    bins = find_bins(frame)
    known_px = known_neon_wavelengths * 1e7
    print(f"Found {len(bins)} bins in calibration file")

    wavelength_arrays, instrument_functions, residuals = {}, {}, {}

    for bin_idx, (lo, hi, _) in enumerate(bins):
        lineout = get_lineout(frame, lo, hi)
        pks, _ = find_peaks(lineout, distance=5, height=lineout.max() * 0.05)
        if len(pks) < 5:
            print(f"  Bin {bin_idx}: only {len(pks)} peaks found, skipping"); continue

        top5_px = np.sort(pks[np.argsort(lineout[pks])[-5:]])
        coeffs = np.polyfit(top5_px, known_px, deg=2)
        wl_axis = np.polyval(coeffs, np.arange(W))
        wavelength_arrays[bin_idx] = wl_axis
        bin_residuals = np.polyval(coeffs, top5_px) - known_px
        residuals[bin_idx] = bin_residuals
        print(f"  Bin {bin_idx}: {wl_axis[0]:.2f}–{wl_axis[-1]:.2f} nm  residuals: {bin_residuals}")

        sigmas, gammas, fits = fit_voigt(lineout, top5_px)

        if plot:
            n = len(top5_px)
            fig, axes = plt.subplots(2, n, figsize=(max(12, 3*n), 6))
            ax_top = fig.add_subplot(2, n, (1, n))
            for a in axes[0]: a.set_visible(False)
            ax_top.plot(wl_axis, lineout, lw=1)
            for px, wl in zip(top5_px, known_px):
                ax_top.axvline(wl, color='steelblue', lw=0.8, ls='--', alpha=0.7)
                ax_top.plot(wl_axis[px], lineout[px], 'o', color='orange', ms=5)
            ax_top.set_xlabel('Wavelength (nm)'); ax_top.set_ylabel('Intensity')
            ax_top.set_title(f'Bin {bin_idx} — neon calibration');

            for i, ((lo_w, hi_w, popt), wl_center) in enumerate(zip(fits, known_px)):
                ax = axes[1, i]
                x = np.arange(lo_w, hi_w, dtype=float)
                ax.plot(x, lineout[lo_w:hi_w], lw=1)
                if popt is not None:
                    x_fine = np.linspace(x[0], x[-1], 500)
                    ax.plot(x_fine, (lambda x, a, c, s, g, b: a*voigt_profile(x-c,s,g)+b)(x_fine, *popt), color='red', lw=1.5)
                    ax.set_title(f'{wl_center:.2f} nm\nσ={popt[2]:.2f} γ={popt[3]:.2f}', fontsize=8)
                else:
                    ax.set_title(f'{wl_center:.2f} nm\nfit failed', fontsize=8, color='red')
                if i == 0: ax.set_ylabel('Intensity')
                if i != 0: ax.set_yticklabels([])
            axes[1, n//2].set_xlabel('Pixel')
            plt.tight_layout(); plt.show(block=False); plt.pause(2); plt.close(fig)

        if instrument_func and sigmas:
            dispersion = abs(wl_axis[W//2+1] - wl_axis[W//2])
            s_nm, g_nm = np.mean(sigmas)*dispersion, np.mean(gammas)*dispersion
            fwhm = 0.5346*2*g_nm + np.sqrt(0.2166*(2*g_nm)**2 + (2.355*s_nm)**2)
            instrument_functions[bin_idx] = {'sigma_nm': s_nm, 'gamma_nm': g_nm, 'fwhm_nm': fwhm, 'dispersion_nm_per_px': dispersion}
            print(f"  Bin {bin_idx}: σ={s_nm:.4f} nm  γ={g_nm:.4f} nm  FWHM≈{fwhm:.4f} nm")

    return CalibrationResult(
        wavelength_arrays=wavelength_arrays,
        instrument_functions=instrument_functions,
        frame=frame,
        bins=bins,
        residuals=residuals
    )

def recalibrate(cal: CalibrationResult, bin_indices, known_neon_wavelengths=KNOWN_NEON_WL):
    known_px = known_neon_wavelengths * 1e7
    W = cal.frame.shape[1]
    n_peaks = len(known_neon_wavelengths)

    for bin_idx in bin_indices:
        lo, hi, _ = cal.bins[bin_idx]
        lineout = get_lineout(cal.frame, lo, hi)

        print(f"Bin {bin_idx}: click {n_peaks} peaks in wavelength order.")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(W), lineout, lw=1)
        ax.set_title(f"Bin {bin_idx} — 0/{n_peaks} clicked")
        plt.tight_layout()

        clicked_px = []
        def on_click(event, clicked_px=clicked_px):
            if event.inaxes != ax or event.button != 1: return
            cx = int(round(event.xdata))
            window = 10
            lo_w, hi_w = max(0, cx-window), min(W, cx+window)
            cx = lo_w + np.argmax(lineout[lo_w:hi_w])
            clicked_px.append(cx)
            ax.axvline(cx, color='red', lw=1, ls='--')
            ax.set_title(f"Bin {bin_idx} — {len(clicked_px)}/{n_peaks} clicked")
            fig.canvas.draw()
            if len(clicked_px) == n_peaks:
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

        coeffs = np.polyfit(np.array(clicked_px), known_px, deg=2)
        cal.wavelength_arrays[bin_idx] = np.polyval(coeffs, np.arange(W))
        cal.residuals[bin_idx] = np.polyval(coeffs, np.array(clicked_px)) - known_px
        print(f"Bin {bin_idx} residuals: {cal.residuals[bin_idx]}")

        sigmas, gammas, fits = fit_voigt(lineout, clicked_px)
        fig2, axes = plt.subplots(1, n_peaks, figsize=(3*n_peaks, 3))
        if n_peaks == 1: axes = [axes]
        for i, ((lo_w, hi_w, popt), wl_center) in enumerate(zip(fits, known_px)):
            x = np.arange(lo_w, hi_w, dtype=float)
            axes[i].plot(x, lineout[lo_w:hi_w], lw=1)
            if popt is not None:
                x_fine = np.linspace(x[0], x[-1], 500)
                axes[i].plot(x_fine, (lambda x,a,c,s,g,b: a*voigt_profile(x-c,s,g)+b)(x_fine,*popt), color='red', lw=1.5)
                axes[i].set_title(f'{wl_center:.2f} nm\nσ={popt[2]:.2f} γ={popt[3]:.2f}', fontsize=8)
            else:
                axes[i].set_title(f'{wl_center:.2f} nm\nfit failed', fontsize=8, color='red')
            if i != 0: axes[i].set_yticklabels([])
        axes[0].set_ylabel('Intensity')
        axes[n_peaks//2].set_xlabel('Pixel')
        plt.tight_layout(); plt.show()

    return cal


if __name__ == '__main__':
    main()
