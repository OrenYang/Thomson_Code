import sys
import types

# create a module alias so pickle can find the old class name
sys.modules['thomson_analysis_preprocess'] = sys.modules[__name__]
"""
Classes
-------

Thomson
    Loads and processes a Thomson scattering .sif file.

    Attributes
    ----------
    name     : str         - filename without extension
    shot     : str         - shot number from filename
    data     : np.ndarray  - raw 2D CCD intensity array
    bins     : list[Bin]   - fiber lineouts; set by lineout()
    cal      : Calfile     - calibration object; set by calibrate()

    Methods
    -------
    __init__(sif_file, name, plot)         - load .sif file
    lineout(bin_ranges, fiber_nums,        - extract 1D spectra per fiber
            manual, plot)
    redo_background(fiber, plot)           - resubract background interactively
    calibrate(cal, plot)                   - apply wavelength calibration
    remove_bins(fibers)                    - delete fibers from self.bins
    plot()                                 - show image and/or lineouts
    save(folder, name)                     - pickle object to disk


Bin  (dataclass)
    One fiber's spectral data.

    Attributes
    ----------
    fiber     : int        - fiber number
    range     : tuple      - (lo, hi) row indices in raw image
    spectrum  : np.ndarray - background-subtracted intensity
    px        : np.ndarray - pixel indices for spectrum
    wavelength: np.ndarray - nm per pixel; set by calibrate()
    pks       : np.ndarray - neon peak indices; set by Calfile
    s, g      : float      - mean Voigt sigma/gamma in meters; set by instrument_function()
    s_std,    : float      - std of sigma/gamma across peaks
    g_std
    s_list,   : list       - per-peak Voigt fit params
    g_list,
    a_list
    xs, ys    : list       - pixel/intensity arrays used for each peak fit
    dispersion: float      - wavelength per pixel (m/px)


Calfile
    Loads a neon calibration .sif and builds a pixel→wavelength map per fiber.

    Attributes
    ----------
    name  : str        - filename without extension
    shot  : str        - shot number from filename
    data  : np.ndarray - raw 2D CCD intensity array
    bins  : list[Bin]  - calibrated fiber bins with wavelength and pks set

    Methods
    -------
    __init__(name, sif_file, plot_sif,          - load .sif, detect neon peaks,
             bin_ranges, fiber_nums, manual,       fit pixel→wavelength polynomial
             plot_lineouts, known_neon_lines)
    instrument_function(plot)                   - fit Voigt profiles to neon peaks;
                                                  populates s, g, dispersion on each bin
    refit_peak(fiber, peak_idx, p0, window,     - refit one peak and update bin params
               plot)
    redo_background(fiber, window, plot)        - interactively redo background,
                                                  re-detect peaks, refit all Voigts
    rebin(fiber, manual, bin_ranges,            - re-extract spatial bin(s)
          fiber_nums, plot)
    plot()                                      - show image and lineouts
    plot_instrument_function(fiber, bin,        - show spectrum + Voigt fits per fiber
                             blocking)
    remove_bins(fibers)                         - delete fibers from self.bins
    _compute_instrument_params(bin)             - internal: recompute s, g from stored lists
"""

import sif_parser
import matplotlib
matplotlib.use("TkAgg")
from tkinter import filedialog, Tk
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pickle
import os
import ast
import csv

import astropy.units as u
from lmfit import Parameters
from plasmapy.diagnostics import thomson

known_neon_lines = np.array([522.23519, 523.40271, 527.40393, 528.00853, 529.81891])  # nm

DEFAULT_CONFIG = {
    # Experimental setup
    "probe_wavelength":     526.5 * u.nm,
    "probe_vec":            np.array([1, 0, 0]),
    "scattering_angle":     np.deg2rad(90),
    "method":               "differential_evolution",

    "ions":                 ['Ne-20 +8', 'Ne-20 +2', 'Ne-20 0+'],

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

    "single_ion_speed":     False,
    "notch_half_width":     None,   # nm
}

def main():
    test=Thomson('/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/TS_analysis/Preprocessed/07234_TS_shotn.pkl')
    test.fit([4,5,8])


class Thomson:
    def __init__(self, file=None, name=None, plot=False):
        if file is None:
            file = filedialog.askopenfilename(
                title="Select a .sif or .pkl file",
                filetypes=[("All supported", "*.sif *.pkl"), ("SIF files", "*.sif"), ("Pickle files", "*.pkl")]
            )

        ext = os.path.splitext(file)[1].lower()

        if ext == '.pkl':
            with open(file, 'rb') as f:
                obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            print(f'Loaded {self.name} from pkl')

        elif ext == '.sif':
            if name is None:
                self.name = file.split('/')[-1].split('.')[0]
                self.shot = self.name.split('_')[0]
            data, info = sif_parser.np_open(file)
            self.data = data[0]
            print(f'Loaded Shot {self.shot}; {self.name}')
            if plot:
                plt.imshow(self.data, cmap='inferno')
                plt.show()

        else:
            raise ValueError(f"Unsupported file type: {ext}. Expected .sif or .pkl")

    def lineout(self, bin_ranges=None, fiber_nums=None, manual=False, plot=False):
        '''
        Automatically detects bins by taking a vertical lineout and finding
        the peaks (which correspond to the light from a fiber) and then moving from
        each peak to find the minimum on either side (corresponding to the dark point between fibers)
        It then averages vertically across the bin to create a lineout for each fiber,
        which is stored as a Bin object
        '''
        profile = np.sum(self.data, axis=1)
        if manual:
            bin_ranges = _manual_bin_selection(profile, self.data)
        _create_lineout(self, profile, bin_ranges, fiber_nums, plot)

    def redo_background(self, fiber=None, plot=False):
        if fiber is None:
            bins_to_process = self.bins
        else:
            bins_to_process = [next(b for b in self.bins if b.fiber == fiber)]

        for b in bins_to_process:
            raw = np.sum(self.data[b.range[0]:b.range[1]], axis=0)
            px_full = np.arange(len(raw))
            b.px, b.spectrum = _manual_background_selection(raw, px_full)

        if plot:
            _plot_lineout(self)

    def calibrate(self, cal, plot=False):
        self.cal = cal
        for i in self.bins:
            for j in self.cal.bins:
                if i.fiber == j.fiber:
                    interp = interp1d(j.px,j.wavelength)
                    i.wavelength =  interp(i.px)
            if plot:
                plt.plot(i.wavelength,i.spectrum)
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.title(f'Fiber {i.fiber}')
                plt.show(block=False)
                plt.pause(1)
                plt.close()

    def set_pin_pos(self, pin_cal, fiber_spacing_um=250):
        """
        Set radial position (in um) on each bin based on a pin position
        calibration shot. The fiber whose bin range best matches the pin
        peak is set to r=0; fibers are spaced fiber_spacing_um apart based
        on their fiber number relative to that reference.

        pin_cal:           a PinCal object
        fiber_spacing_um:  spacing between adjacent fiber centers, in microns.
                            Pass a different value if your setup changes.
        """
        bin_ranges = [b.range for b in self.bins]
        axis_idx, axis_range = pin_cal.find_axis_range(bin_ranges)
        axis_fiber = self.bins[axis_idx].fiber

        for b in self.bins:
            b.r = (b.fiber - axis_fiber) * fiber_spacing_um

        print(f"Set r=0 at fiber {axis_fiber}. Spacing = {fiber_spacing_um} um/fiber.")
        for b in sorted(self.bins, key=lambda x: x.fiber):
            print(f"  Fiber {b.fiber}: r = {b.r:+.0f} um")

    def remove_bins(self, fibers):
        if isinstance(fibers, int):
            fibers = [fibers]
        self.bins = [b for b in self.bins if b.fiber not in fibers]
        print(f"Removed fiber(s) {fibers}. Remaining: {[b.fiber for b in self.bins]}")

    def plot(self, type='result', fiber=None, block=False):
        if type.strip().lower() == 'summary':
            print('summary')

        elif type.strip().lower() == 'result' and hasattr(self.bins[0], 'result'):
            if isinstance(fiber, int):
                fiber = [fiber]
            if fiber is not None:
                if len(fiber) < 4:
                    block=True
            for b in self.bins:
                if fiber is None or b.fiber in fiber:
                    if not hasattr(b, 'result'):
                        print(f"Fiber {b.fiber}: no fit result yet.")
                        continue
                    cfg = {**DEFAULT_CONFIG, **(b.config or {})}
                    wavelengths = b.wavelength
                    fig, ax = plt.subplots()
                    ax.set_xlabel("Wavelength (nm)")
                    ax.set_ylabel("Skw")
                    ax.set_title(f"Fiber {b.fiber}")
                    ax.axvline(x=cfg['probe_wavelength'].value, color="red", label="Probe wavelength")
                    ax.set_xlim(523, 530)
                    ax.plot(wavelengths, b.spectrum / b.spectrum.max(), label="Data")
                    ax.plot(wavelengths, b.result['best_fit'], label="Best-fit")
                    ax.legend(loc="upper right")
                    plt.tight_layout()
                    plt.show(block=block)
                    if not block:
                        plt.pause(1)
                        plt.close()

        elif type.strip().lower() == 'sif' or not hasattr(self, 'bins'):
            plt.imshow(self.data, cmap='inferno')
            plt.title(f'{self.name} (no lineout computed yet)')
            plt.xlabel('px')
            plt.ylabel('px')
            plt.colorbar()
            plt.show(block=False)

        else:
            _plot_lineout(self)

    def fit(self, fiber=None, config=None):
        if not hasattr(self, "cal"):
            raise Exception("The data must be calibrated before fitting")
        if isinstance(fiber, int):
            fiber = [fiber]
        for i in self.bins:
            cal_bin = next(c for c in self.cal.bins if c.fiber == i.fiber)
            if not hasattr(i, 'config'):
                i.config = {}
            i.config["sigma"] = cal_bin.s
            i.config["gamma"] = cal_bin.g
            if fiber is None or i.fiber in fiber:
                try:
                    iaw_result = fit_iaw(i.spectrum, i.wavelength, config=i.config)
                    i.result = {
                        'best_fit':   iaw_result.best_fit,
                        'params':     {k: (p.value, p.stderr) for k, p in iaw_result.params.items()},
                        'chisqr':     iaw_result.chisqr,
                        'redchi':     iaw_result.redchi,
                        'rsquared':   iaw_result.rsquared,
                        'fit_report': iaw_result.fit_report(),
                    }
                except Exception as e:
                    print(f'Skipping Fiber {i.fiber}: {e}')

    def add_config(self, config, fiber=None):
        if isinstance(fiber, int):
            fiber = [fiber]
        for i in self.bins:
            if fiber is None or i.fiber in fiber:
                if not hasattr(i, 'config'):
                    i.config = {}
                i.config.update(config)

    def update_config(self, fiber=None, field=None, value=None):
        if isinstance(fiber, int):
            fiber = [fiber]

        if field is None:
            new_config = make_config()
            for b in self.bins:
                if fiber is None or b.fiber in fiber:
                    if not hasattr(b, 'config'):
                        b.config = {}
                    b.config.update(new_config)
            return

        if field not in DEFAULT_CONFIG:
            print(f"Unknown field '{field}'. Valid fields: {list(DEFAULT_CONFIG.keys())}")
            return

        default = DEFAULT_CONFIG[field]
        VALID_METHODS = ['differential_evolution', 'leastsq', 'nelder', 'powell', 'cobyla']

        if value is None:
            while True:
                try:
                    if field == 'scattering_angle':
                        val = input(f"{field} (deg) [default: {np.rad2deg(default):.4g}]: ").strip()
                        value = np.deg2rad(float(val)) if val else default
                    elif field == 'method':
                        val = input(f"{field} [default: {default}] options: {VALID_METHODS}: ").strip()
                        if val and val not in VALID_METHODS:
                            raise ValueError(f"Must be one of {VALID_METHODS}")
                        value = val if val else default
                    elif isinstance(default, list) and isinstance(default[0], str):
                        val = input(f"{field} [default: {default}]: ").strip()
                        parsed = ast.literal_eval(val) if val else default
                        if val:
                            if not isinstance(parsed, list):
                                raise ValueError("Expected a list")
                            from plasmapy.particles import Particle
                            for p in parsed:
                                Particle(p)
                        value = parsed
                    elif isinstance(default, u.Quantity):
                        val = input(f"{field} [default: {default}]: ").strip()
                        value = float(val) * default.unit if val else default
                    elif isinstance(default, np.ndarray):
                        val = input(f"{field} [default: {default.tolist()}]: ").strip()
                        value = np.array(ast.literal_eval(val)) if val else default
                    elif isinstance(default, list):
                        val = input(f"{field} [default: {default}]: ").strip()
                        parsed = ast.literal_eval(val) if val else default
                        if val and not isinstance(parsed, list):
                            raise ValueError("Expected a list")
                        value = parsed
                    elif isinstance(default, bool):
                        val = input(f"{field} [default: {default}]: ").strip()
                        if val and val.lower() not in ('true', 'false'):
                            raise ValueError("Expected 'true' or 'false'")
                        value = (val.lower() == 'true') if val else default
                    elif default is None:
                        val = input(f"{key} [default: None]: ").strip()
                        config[key] = float(val) if val else None
                    else:
                        val = input(f"{field} [default: {default}]: ").strip()
                        value = type(default)(val) if val else default
                    break
                except Exception as e:
                    print(f"  Invalid input: {e}. Try again.")

        for b in self.bins:
            if fiber is None or b.fiber in fiber:
                if not hasattr(b, 'config'):
                    b.config = {}
                b.config[field] = value

    def print_results(self, fiber=None, save_csv=True, csv_path=None):
        if isinstance(fiber, int):
            fiber = [fiber]

        rows = []
        for b in self.bins:
            if fiber is None or b.fiber in fiber:
                if not hasattr(b, 'result'):
                    print(f"Fiber {b.fiber}: no fit result yet.")
                    continue
                print(f"\n=== Fiber {b.fiber} ===")
                print(b.result['fit_report'])

                cfg = {**DEFAULT_CONFIG, **(b.config or {})}
                ions = cfg.get('ions', [])

                row = {'fiber': b.fiber, 'r_um': getattr(b, 'r', ''),
                       'chisqr': b.result['chisqr'],
                       'redchi': b.result['redchi'], 'rsquared': b.result['rsquared']}

                for key, (val, err) in b.result['params'].items():
                    # rename T_i_0 -> T_i_0_NeIon, ifract_0 -> ifract_0_NeIon, etc.
                    renamed_key = key
                    if key.startswith(('T_i_', 'ion_speed_', 'ifract_', 'sfract_')):
                        try:
                            idx = int(key.rsplit('_', 1)[-1])
                            if idx < len(ions):
                                species = ions[idx].replace(' ', '')
                                renamed_key = f'{key}_{species}'
                        except (ValueError, IndexError):
                            pass
                    row[renamed_key] = val
                    row[f'{renamed_key}_err'] = err if err is not None else ''
                rows.append(row)

        if save_csv and rows:
            if csv_path is None:
                csv_path = f'{self.name}_results.csv'

            fieldnames = list(rows[0].keys())
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)

            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nSaved results to {csv_path}")

    def save(self, folder=None, name=None):
        if name is None:
            name=self.name
        if folder is None:
            folder = 'TS_spectra'
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f'Saved {self.name} to {path}')


@dataclass
class Bin:
    fiber: int
    range: tuple
    spectrum: np.array
    px: np.array

class Calfile:
    def __init__(self,name = None,sif_file=None,plot_sif=False, bin_ranges=None, fiber_nums=None, manual=False, plot_lineouts=False, known_neon_lines=np.array([522.23519, 523.40271, 527.40393, 528.00853, 529.81891])):
        if sif_file is None:
            sif_file = filedialog.askopenfilename(title="Select a .sif file")
        if name is None:
            self.name = sif_file.split('/')[-1].split('.')[0]
            self.shot = self.name.split('_')[0]
        data, info = sif_parser.np_open(sif_file)
        self.data = data[0]
        print(f'Loaded Calibration from shot {self.shot}; {self.name}')
        if plot_sif:
            plt.imshow(self.data,cmap='inferno')
            plt.show()
        profile = np.sum(self.data,axis=1)
        if manual:
            bin_ranges = _manual_bin_selection(profile, self.data)
        _create_lineout(self, profile, bin_ranges, fiber_nums, plot=plot_lineouts, subtract='cal')
        for bin in self.bins:
            bin.pks, _ = find_peaks(bin.spectrum, width=4,height=0.12*bin.spectrum.max())
            if len(bin.pks) != 5:
                bin.pks = _manual_peak_selection(bin)
            coefs = np.polyfit(bin.pks[:5],known_neon_lines,2)
            bin.wavelength = np.polyval(coefs,bin.px)
            if plot_lineouts:
                plt.plot(bin.wavelength,bin.spectrum)
                plt.scatter(bin.wavelength[bin.pks],bin.spectrum[bin.pks],color='red')
                plt.title(f'Fiber {bin.fiber}')
                plt.show(block=False)
                plt.pause(1)
                plt.close()

    def instrument_function(self, plot=False):
        for bin in self.bins:
            xs = []
            ys = []
            bin.s_list = []
            bin.g_list = []
            bin.a_list = []

            for pk in bin.pks:
                xs.append(bin.px[max(pk-50,0):min(pk+50,len(bin.px))].copy() - bin.px[pk])
                ys.append(bin.spectrum[max(0,pk-50):min(len(bin.spectrum),pk+50)].copy())
                ys[-1] /= ys[-1].max()
                p0 = [1.0, 1.0, 1.0]
                popt, _ = curve_fit(lambda x,s,g,a: a*voigt_profile(x,s,g), xs[-1], ys[-1], p0)
                bin.s_list.append(popt[0])
                bin.g_list.append(popt[1])
                bin.a_list.append(popt[2])

            bin.xs = xs  # store for later refitting
            bin.ys = ys

            dispersion = np.mean(np.diff(bin.wavelength)) * 1e-9
            bin.dispersion = dispersion
            self._compute_instrument_params(bin)

            if plot:
                self.plot_instrument_function(bin=bin, blocking=False)

    def refit_peak(self, fiber, peak_idx, p0=None, window=50, plot=True):
        """
        Refit a single peak and update the instrument function parameters.
        fiber:     fiber number
        peak_idx:  index into bin.pks (0 to 4)
        p0:        optional initial guess [sigma, gamma, amplitude]
        """
        bin = next(b for b in self.bins if b.fiber == fiber)

        if not hasattr(bin, 's_list'):
            raise RuntimeError("Run instrument_function() before refitting peaks.")

        pk = bin.pks[peak_idx]
        x = bin.px[max(pk-window,0):min(pk+window,len(bin.px))].copy() - bin.px[pk]
        y = bin.spectrum[max(0,pk-window):min(len(bin.spectrum),pk+window)].copy()
        y /= y.max()

        if p0 is None:
            p0 = [1.0, 1.0, 1.0]

        popt, _ = curve_fit(lambda x,s,g,a: a*voigt_profile(x,s,g), x, y, p0)
        s, g, a = popt

        # Update stored lists
        bin.s_list[peak_idx] = s
        bin.g_list[peak_idx] = g
        bin.a_list[peak_idx] = a
        bin.xs[peak_idx] = x
        bin.ys[peak_idx] = y

        # Recompute means
        self._compute_instrument_params(bin)

        if plot:
            self.plot_instrument_function(bin)

        print(f"Fiber {fiber}, peak {peak_idx}: s={s:.4f}, g={g:.4f}, a={a:.4f}")
        print(f"Updated bin.s={bin.s:.4e}, bin.g={bin.g:.4e}")

    def redo_background(self, fiber, window=50, plot=True):
        bin = next(b for b in self.bins if b.fiber == fiber)

        raw = np.sum(self.data[bin.range[0]:bin.range[1]], axis=0)
        px_full = np.arange(len(raw))

        # get background region from user
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(px_full, raw)
        ax.set_title("Click LEFT and RIGHT edges of the background region.")
        ax.set_xlabel('px')
        clicks = []
        def onclick(event):
            if event.inaxes and len(clicks) < 2:
                clicks.append(int(round(event.xdata)))
                ax.axvline(event.xdata, color='red', linestyle='--')
                fig.canvas.draw()
                if len(clicks) == 2:
                    plt.close(fig)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

        bg_lo, bg_hi = sorted(clicks)
        background = np.mean(raw[bg_lo:bg_hi])

        # keep full px, just subtract background
        bin.px = px_full
        bin.spectrum = raw - background

        # re-detect peaks on full px array
        new_pks, _ = find_peaks(bin.spectrum, width=5, height=0.12*bin.spectrum.max())
        if len(new_pks) != 5:
            new_pks = _manual_peak_selection(bin)
        bin.pks = new_pks

        coefs = np.polyfit(bin.pks[:5], known_neon_lines, 2)
        bin.wavelength = np.polyval(coefs, bin.px)
        bin.dispersion = np.mean(np.diff(bin.wavelength)) * 1e-9

        bin.s_list = []
        bin.g_list = []
        bin.a_list = []
        bin.xs = []
        bin.ys = []
        for pk in bin.pks:
            x = bin.px[max(pk-window,0):min(pk+window,len(bin.px))].copy() - bin.px[pk]
            y = bin.spectrum[max(0,pk-window):min(len(bin.spectrum),pk+window)].copy()
            y /= y.max()
            popt, _ = curve_fit(lambda x,s,g,a: a*voigt_profile(x,s,g), x, y, [1.0, 1.0, 1.0])
            bin.s_list.append(popt[0])
            bin.g_list.append(popt[1])
            bin.a_list.append(popt[2])
            bin.xs.append(x)
            bin.ys.append(y)

        self._compute_instrument_params(bin)
        print(f"Updated bin.s={bin.s:.4e}, bin.g={bin.g:.4e}")

        if plot:
            self.plot_instrument_function(fiber=fiber)

    def rebin(self, fiber=None, manual=True, bin_ranges=None, fiber_nums=None, plot=False):
        profile = np.sum(self.data, axis=1)

        if fiber is not None:
            existing = next((b for b in self.bins if b.fiber == fiber), None)

            if manual:
                print(f"Select new range for fiber {fiber}, then press Enter.")
                bin_range = _manual_bin_selection(profile, self.data)[0]
            else:
                bin_range = bin_ranges[0]

            lo, hi = bin_range
            raw = np.sum(self.data[lo:hi], axis=0)
            px, spectrum = _subtract_background(raw, type='exp')

            if existing is not None:
                existing.range = (lo, hi)
                existing.px = px
                existing.spectrum = spectrum
            else:
                self.bins.append(Bin(fiber, (lo, hi), spectrum, px))
                print(f"Added new fiber {fiber}.")

        else:
            if manual:
                bin_ranges = _manual_bin_selection(profile, self.data)
            _create_lineout(self, profile, bin_ranges, fiber_nums, plot)

        if plot:
            _plot_lineout(self)

    def plot(self):
        if hasattr(self, 'bins'):
            _plot_lineout(self)
        else:
            plt.imshow(self.data, cmap='inferno')
            plt.title(f'{self.name} (no lineout computed yet)')
            plt.xlabel('px')
            plt.ylabel('px')
            plt.colorbar()
            plt.show(block=False)

    def plot_instrument_function(self, fiber=None, bin=None, blocking=True):
        """
        bin:      pass a Bin object directly
        fiber:    pass a fiber number
        neither:  plots all fibers
        blocking: if False, auto-closes after 1s (useful when looping internally)
        """
        if bin is not None:
            bins_to_plot = [bin]
        elif fiber is not None:
            bins_to_plot = [next(b for b in self.bins if b.fiber == fiber)]
        else:
            bins_to_plot = self.bins
            blocking=False

        for b in bins_to_plot:
            plt.figure(figsize=(10, 4))
            plt.subplot(2,1,1)
            plt.plot(b.wavelength, b.spectrum)
            plt.title(f'Fiber {b.fiber}')
            plt.scatter(b.wavelength[b.pks], b.spectrum[b.pks], color='red')
            for i in range(len(b.pks)):
                plt.subplot(2, len(b.pks), i+len(b.pks)+1)
                plt.plot(b.xs[i], b.ys[i])
                plt.plot(b.xs[i], b.a_list[i]*voigt_profile(b.xs[i], b.s_list[i], b.g_list[i]), 'r--')
                plt.yticks([])
            if blocking:
                plt.show(block=True)
            else:
                plt.show(block=False)
                plt.pause(1)
                plt.close()

    def remove_bins(self, fibers):
        if isinstance(fibers, int):
            fibers = [fibers]
        self.bins = [b for b in self.bins if b.fiber not in fibers]
        print(f"Removed fiber(s) {fibers}. Remaining: {[b.fiber for b in self.bins]}")

    def _compute_instrument_params(self, bin):
        dispersion = bin.dispersion
        bin.s = np.mean(bin.s_list) * dispersion
        bin.g = np.mean(bin.g_list) * dispersion
        bin.s_std = np.std(bin.s_list) * dispersion
        bin.g_std = np.std(bin.g_list) * dispersion

class PinPos:
    """
    Loads a pin-position calibration .sif file (taken with a pin on-axis,
    illuminating only the fiber at r=0). Used to set radial positions for
    all other fibers based on a fixed fiber spacing.
    """
    def __init__(self, sif_file=None, name=None, plot=False):
        if sif_file is None:
            sif_file = filedialog.askopenfilename(title="Select a pin position .sif file")
        if name is None:
            self.name = sif_file.split('/')[-1].split('.')[0]
            self.shot = self.name.split('_')[0]
        data, info = sif_parser.np_open(sif_file)
        self.data = data[0]
        print(f'Loaded pin position calibration from shot {self.shot}; {self.name}')

        profile = np.sum(self.data, axis=1)
        self.profile = profile

        if plot:
            plt.imshow(self.data, cmap='inferno')
            plt.title('Pin position calibration')
            plt.show()

    def find_axis_range(self, bin_ranges):
        """
        Given a list of (lo, hi) bin ranges (e.g. from an existing Thomson
        object's bins), find which one best matches the lit-up pin fiber.
        Returns (index_into_bin_ranges, (lo, hi)).
        """
        peak_profile = gaussian_filter1d(self.profile, 2)
        pk = np.argmax(peak_profile)

        best_idx = None
        best_dist = np.inf
        for i, (lo, hi) in enumerate(bin_ranges):
            mid = (lo + hi) / 2
            dist = abs(mid - pk)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        lo, hi = bin_ranges[best_idx]
        print(f"Pin peak at px {pk}. Closest bin range: index {best_idx}, range ({lo}, {hi})")
        return best_idx, (lo, hi)

    def plot(self):
        plt.figure()
        plt.plot(self.profile)
        peak_profile = gaussian_filter1d(self.profile, 2)
        pk = np.argmax(peak_profile)
        plt.axvline(pk, color='red', linestyle='--', label=f'Pin peak (px {pk})')
        plt.xlabel('px')
        plt.ylabel('Intensity')
        plt.title(f'{self.name} pin position profile')
        plt.legend()
        plt.show(block=False)

def _find_bins(profile):
    peak_profile=gaussian_filter1d(profile,2)
    peaks, _ = find_peaks(peak_profile, distance=10, height=profile.max() * 0.1)
    bin_ranges=[]

    for pk in peaks:
        lo = pk
        hi = pk
        while lo>0 and peak_profile[lo] > peak_profile[lo-1]:
            lo-=1
        while hi<len(peak_profile)-1 and peak_profile[hi] > peak_profile[hi+1]:
            hi+=1
        bin_ranges.append((lo,hi))
    return bin_ranges

def _subtract_background(spectrum, type, manual=False):
    base = np.mean(spectrum[:50])
    px = np.arange(0, len(spectrum), 1)
    if type == 'exp':
        if manual:
            return _manual_background_selection(spectrum, px)
        for i in range(len(spectrum)):
            temp = spectrum[i:i+50]
            temp2 = np.mean(temp)
            background = np.mean(spectrum[i+50:i+150])
            if temp2 > 1.1*base and min(temp)/max(temp) > 0.95:
                break
        lo=0
        while spectrum[lo] < background:
            lo+=1
        hi=len(spectrum)-1
        while spectrum[hi] < background:
            hi-=1
        spectrum = spectrum[lo:hi]-background
        px = px[lo:hi]
    elif type == 'cal':
        if manual:
            return _manual_background_selection(spectrum, px)
        pks, _ = find_peaks(spectrum, height=0.3*max(spectrum))
        background = np.mean(spectrum[pks[1]+50:pks[1]+150])
        spectrum = spectrum - background
    return px, spectrum

def _manual_background_selection(spectrum, px):
    """Click two points to define spectral range, then two points for background region."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(px, spectrum)
    ax.set_title("Click LEFT and RIGHT edges of the spectrum to keep, then\n"
                 "click LEFT and RIGHT edges of the background region.")
    ax.set_xlabel('px')

    clicks = []
    vlines = []

    def onclick(event):
        if event.inaxes and len(clicks) < 4:
            clicks.append(int(round(event.xdata)))
            color = 'green' if len(clicks) <= 2 else 'red'
            vlines.append(ax.axvline(event.xdata, color=color, linestyle='--', alpha=0.8))
            if len(clicks) == 2:
                ax.set_title("Now click LEFT and RIGHT edges of the background region.",
                             color='red')
            fig.canvas.draw()
            if len(clicks) == 4:
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    if len(clicks) != 4:
        raise ValueError(f"Expected 4 clicks, got {len(clicks)}")

    spec_lo, spec_hi = sorted(clicks[:2])
    bg_lo, bg_hi = sorted(clicks[2:])

    background = np.mean(spectrum[bg_lo:bg_hi])
    mask = (px >= spec_lo) & (px <= spec_hi)
    return px[mask], spectrum[mask] - background

def _create_lineout(obj, profile, bin_ranges, fiber_nums, plot, subtract='exp'):
    if bin_ranges is None:
        bin_ranges = _find_bins(profile)

    if fiber_nums is None or len(fiber_nums) != len(bin_ranges):
        fiber_nums = np.arange(0, len(bin_ranges), 1)

    obj.bins = []

    for i in range(len(bin_ranges)):
        bin_array = np.sum(obj.data[bin_ranges[i][0]:bin_ranges[i][1]], axis=0)
        px, bin_array = _subtract_background(bin_array, type=subtract)
        obj.bins.append(Bin(fiber_nums[i], bin_ranges[i], bin_array, px))

    if plot:
        _plot_lineout(obj)

def _manual_peak_selection(bin):
    fig, ax = plt.subplots()
    ax.plot(bin.px, bin.spectrum)
    ax.set_title("Auto-detection failed. Click near 5 neon peaks, then close.")

    clicked_x = []

    def onclick(event):
        if event.inaxes and len(clicked_x) < 5:
            clicked_x.append(int(round(event.xdata)))
            ax.axvline(event.xdata, color='orange', linestyle='--', alpha=0.7)
            fig.canvas.draw()
            if len(clicked_x) == 5:
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(clicked_x) != 5:
        raise ValueError(f"Expected 5 clicks, got {len(clicked_x)}")

    window = 10
    peaks = []
    for cx in sorted(clicked_x):
        # cx is a pixel index value, find its position in bin.px
        idx = np.argmin(np.abs(bin.px - cx))
        lo = max(0, idx - window)
        hi = min(len(bin.spectrum), idx + window)
        local_max = lo + np.argmax(bin.spectrum[lo:hi])
        peaks.append(local_max)

    return np.array(peaks)

def _plot_lineout(obj):
    bin_ranges = [b.range for b in obj.bins]
    fiber_nums = [b.fiber for b in obj.bins]
    cmap = plt.get_cmap('tab10')

    fig, (ax_img, ax_line) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={'width_ratios': [1, 1]}
    )

    ax_img.imshow(obj.data, cmap='inferno',aspect='auto')

    for i, bin in enumerate(obj.bins):
        color = cmap(i % 10)
        lo, hi = bin_ranges[i]
        mid = (lo + hi) / 2

        ax_img.hlines([lo, hi], 0, obj.data.shape[1],
                      colors=color, linestyles='--', linewidth=0.8)
        ax_img.text(4, mid, f'F{fiber_nums[i]}',
                    color=color, fontsize=8, va='center', fontweight='bold')

        norm = bin.spectrum / (bin.spectrum.max() if bin.spectrum.max() != 0 else 1)
        scale = (hi - lo) * 0.9

        calibrated = hasattr(bin, 'wavelength')
        x = bin.wavelength if calibrated else bin.px
        ax_line.plot(x, obj.data.shape[0] - mid + norm * scale - (hi - lo) / 2,
                     color=color, label=f'F{fiber_nums[i]}')

    ax_img.set_title('Fiber bins')
    ax_img.set_xlabel('px')
    ax_img.set_ylabel('px')

    ax_line.set_ylim(ax_img.get_ylim())
    ax_line.set_yticks([])
    ax_line.invert_yaxis()
    ax_line.set_title('Lineouts')
    ax_line.set_xlabel('Wavelength [nm]' if calibrated else 'px')
    # only lock x to image pixels when uncalibrated
    if not calibrated:
        ax_line.set_xlim(ax_img.get_xlim())
    ax_line.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show(block=False)

def _manual_bin_selection(profile, image):
    fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(12, 6),
                                           gridspec_kw={'width_ratios': [1, 1]})
    ax_img.imshow(image, cmap='inferno', aspect='auto')
    ax_img.set_title('Click lo then hi for each bin.\nPress Enter or close window when finished.')
    ax_img.set_xlabel('px')
    ax_img.set_ylabel('px')

    ax_prof.plot(profile, np.arange(len(profile)), color='white', lw=1)
    ax_prof.set_ylim(len(profile), 0)
    ax_prof.set_facecolor('#1a1a1a')
    ax_prof.set_title('Vertical profile')
    ax_prof.set_xlabel('Intensity')
    ax_prof.set_ylabel('px')

    clicks = []
    bin_ranges = []
    cmap = plt.get_cmap('tab10')

    def onclick(event):
        if event.inaxes not in [ax_img, ax_prof]:
            return
        y = int(round(event.ydata))
        clicks.append(y)
        color = cmap((len(bin_ranges)) % 10)

        ax_img.axhline(y, color=color, linestyle='--', linewidth=0.9, alpha=0.8)
        ax_prof.axhline(y, color=color, linestyle='--', linewidth=0.9, alpha=0.8)

        if len(clicks) % 2 == 0:
            lo, hi = sorted(clicks[-2:])
            bin_ranges.append((lo, hi))
            mid = (lo + hi) / 2
            ax_img.axhspan(lo, hi, alpha=0.15, color=color)
            ax_prof.axhspan(lo, hi, alpha=0.15, color=color)
            ax_img.text(4, mid, f'B{len(bin_ranges)-1}',
                        color=color, fontsize=8, va='center', fontweight='bold')
            print(f"  Bin {len(bin_ranges)-1}: rows {lo}–{hi}")

        fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

    if len(bin_ranges) == 0:
        raise RuntimeError("No bins were selected manually. Aborting.")

    print(f"Manual selection complete: {len(bin_ranges)} bin(s) defined.")
    return bin_ranges

def _make_instr_func(sigma,gamma):
    def instrument_function(wavelengths):
        w = wavelengths.to(u.m).value
        instr = voigt_profile(w, sigma, gamma)
        return instr / instr.max()
    return instrument_function

def fit_iaw(iaw_skw, iaw_wavelengths, config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    iaw_skw = iaw_skw / iaw_skw.max()
    iaw_wavelengths = iaw_wavelengths * u.nm

    weights = np.ones_like(iaw_skw)
    notch_quantity = None
    if cfg.get('notch_half_width'):
        probe_wl = cfg['probe_wavelength'].to(u.nm).value
        lo = probe_wl - cfg['notch_half_width']
        hi = probe_wl + cfg['notch_half_width']
        in_notch = (iaw_wavelengths.value >= lo) & (iaw_wavelengths.value <= hi)
        weights[in_notch] = 0.0
        notch_quantity = np.array([lo, hi]) * u.nm

    scatter_vec = np.array([np.cos(cfg['scattering_angle']), np.sin(cfg['scattering_angle']), 0])
    dk = scatter_vec - cfg['probe_vec']
    dk_hat = dk / np.linalg.norm(dk)

    sfracts = []
    remaining = 1.0
    for i in range(len(cfg['ions']) - 1):
        s = cfg['ifract'][i] / remaining
        sfracts.append(s)
        remaining -= cfg['ifract'][i]

    params = Parameters()

    params.add('n', value=cfg['n'], min=cfg['n_min'], max=cfg['n_max'], vary=cfg['n_vary'])

    single_ion_speed = cfg.get('single_ion_speed', False)

    for i in range(len(cfg['ions'])):
        params.add(f'T_i_{i}', value=cfg['T_ion'][i], min=cfg['T_ion_min'][i],
                   max=cfg['T_ion_max'][i], vary=cfg['T_ion_vary'][i])

        if single_ion_speed:
            if i == 0:
                # leader: the one shared floating velocity
                params.add('ion_speed_0', value=cfg['ion_speed'][0],
                           min=cfg['ion_speed_min'][0], max=cfg['ion_speed_max'][0],
                           vary=cfg['ion_speed_vary'][0])
            else:
                # every other ion ties to ion_speed_0
                params.add(f'ion_speed_{i}', expr='ion_speed_0')
        else:
            params.add(f'ion_speed_{i}', value=cfg['ion_speed'][i],
                       min=cfg['ion_speed_min'][i], max=cfg['ion_speed_max'][i],
                       vary=cfg['ion_speed_vary'][i])

    for i in range(len(cfg['ions']) - 1):
        params.add(f'sfract_{i}', value=sfracts[i], min=0, max=1, vary=cfg['ifract_vary'][i])
    for i in range(len(cfg['ions']) - 1):
        if i == 0:
            expr = 'sfract_0'
        else:
            remaining = ' * '.join([f'(1 - sfract_{j})' for j in range(i)])
            expr = f'sfract_{i} * {remaining}'
        params.add(f'ifract_{i}', expr=expr)
    params.add(f'ifract_{len(cfg["ions"])-1}', expr='1 - ' + ' - '.join([f'ifract_{j}' for j in range(len(cfg['ions'])-1)]))
    params.add('T_e_0', value=cfg['T_e'], min=cfg['T_e_min'], max=cfg['T_e_max'], vary=cfg['T_e_vary'])
    params.add('electron_speed_0', value=cfg['e_speed'], min=cfg['e_speed_min'],
               max=cfg['e_speed_max'], vary=cfg['e_speed_vary'])

    settings = {}
    settings["probe_wavelength"] = cfg['probe_wavelength'].to(u.m).value
    settings["probe_vec"] = cfg['probe_vec']
    settings["scatter_vec"] = scatter_vec
    settings["ions"] = cfg['ions']
    settings["ion_vdir"] = np.tile(dk_hat, (len(cfg['ions']), 1))
    settings["electron_vdir"] = dk_hat.reshape(1, 3)
    settings["instr_func"] = _make_instr_func(cfg['sigma'], cfg['gamma'])
    if notch_quantity is not None:
        settings["notch"] = notch_quantity.to(u.m).value

    iaw_model = thomson.spectral_density_model(
        iaw_wavelengths.to(u.m).value,
        settings,
        params,
    )

    iaw_result = iaw_model.fit(
        iaw_skw,
        params=params,
        wavelengths=iaw_wavelengths.to(u.m).value,
        weights=weights,
        method=cfg['method'],
    )

    best_fit_skw = iaw_result.best_fit

    fig, ax = plt.subplots(ncols=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Skw")
    ax.axvline(x=cfg['probe_wavelength'].value, color="red", label="Probe wavelength")
    ax.set_xlim(523, 530)
    ax.plot(iaw_wavelengths.value, iaw_skw, label="Data")
    ax.plot(iaw_wavelengths.value, best_fit_skw, label="Best-fit")
    ax.legend(loc="upper right")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    return iaw_result

def make_config():
    config = {}
    print("Press enter to use default value.")

    for key, default in DEFAULT_CONFIG.items():
        VALID_METHODS = ['differential_evolution', 'leastsq', 'nelder', 'powell', 'cobyla']
        while True:
            try:
                if key == 'scattering_angle':
                    val = input(f"{key} (deg) [default: {np.rad2deg(default)}]: ").strip()
                    config[key] = np.deg2rad(float(val)) if val else default

                elif key == 'method':
                    val = input(f"{key} [default: {default}] options: {VALID_METHODS}: ").strip()
                    if val and val not in VALID_METHODS:
                        raise ValueError(f"Must be one of {VALID_METHODS}")
                    config[key] = val if val else default

                elif isinstance(default, list) and isinstance(default[0], str):
                    val = input(f"{key} [default: {default}]: ").strip()
                    parsed = ast.literal_eval(val) if val else default
                    if val:
                        if not isinstance(parsed, list):
                            raise ValueError("Expected a list")
                        from plasmapy.particles import Particle
                        for p in parsed:
                            Particle(p)  # raises if invalid
                    config[key] = parsed

                elif isinstance(default, u.Quantity):
                    val = input(f"{key} [default: {default}]: ").strip()
                    config[key] = float(val) * default.unit if val else default

                elif isinstance(default, np.ndarray):
                    val = input(f"{key} [default: {default.tolist()}]: ").strip()
                    if val:
                        parsed = np.array(ast.literal_eval(val))
                        if parsed.ndim != default.ndim or parsed.shape[-1] != default.shape[-1]:
                            raise ValueError(f"Expected shape (n, {default.shape[-1]}), got {parsed.shape}")
                        config[key] = parsed
                    else:
                        config[key] = default

                elif isinstance(default, list):
                    val = input(f"{key} [default: {default}]: ").strip()
                    parsed = ast.literal_eval(val) if val else default
                    if val and not isinstance(parsed, list):
                        raise ValueError("Expected a list")
                    config[key] = parsed

                elif isinstance(default, bool):
                    val = input(f"{key} [default: {default}]: ").strip()
                    if val and val.lower() not in ('true', 'false'):
                        raise ValueError("Expected 'true' or 'false'")
                    config[key] = val.lower() == 'true' if val else default

                else:
                    val = input(f"{key} [default: {default}]: ").strip()
                    config[key] = type(default)(val) if val else default

                break

            except Exception as e:
                if isinstance(default, np.ndarray):
                    print(f"  Invalid input: {e}. Expected format: {default.tolist()}")
                elif isinstance(default, list):
                    print(f"  Invalid input: {e}. Expected format: {default}")
                elif isinstance(default, bool):
                    print(f"  Invalid input. Expected 'true' or 'false'.")
                elif key == 'scattering_angle':
                    print(f"  Invalid input. Expected a number in degrees.")
                elif isinstance(default, u.Quantity):
                    print(f"  Invalid input. Expected a number in {default.unit}.")
                else:
                    print(f"  Invalid input. Expected type: {type(default).__name__}.")

    return config

def load_config(config_file=None):
    if config_file is None:
        config_file = filedialog.askopenfilename(
            title="Select a config file",
            filetypes=[("Text files", "*.txt")]
        )
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            if key not in DEFAULT_CONFIG:
                print(f"  Warning: unknown key '{key}', skipping")
                continue
            default = DEFAULT_CONFIG[key]
            if key == 'scattering_angle':
                config[key] = np.deg2rad(float(val))
            elif isinstance(default, u.Quantity):
                config[key] = float(val) * default.unit
            elif isinstance(default, np.ndarray):
                config[key] = np.array(ast.literal_eval(val))
            elif isinstance(default, (list, bool)):
                config[key] = ast.literal_eval(val)
            else:
                config[key] = type(default)(val)
    return {**DEFAULT_CONFIG, **config}

if __name__ == "__main__":
    main()
