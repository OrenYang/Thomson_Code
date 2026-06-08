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


known_neon_lines = np.array([522.23519, 523.40271, 527.40393, 528.00853, 529.81891])  # nm

def main():
    test=Thomson(plot=False)
    test.lineout(manual=False,plot=False)
    cal=Calfile(plot_sif=False, plot_lineouts=False)
    cal.instrument_function(plot=False)
    test.calibrate(cal,plot=False)
    test.plot_lineout()


class Thomson:
    def __init__(self, sif_file=None, name=None, plot=False):
        '''
        Loads .sif file to create a Thomson object
        Initiallizes with the 2D array of intensities from the sif, the shot number, and the file name
        '''
        if sif_file is None:
            sif_file = filedialog.askopenfilename(title="Select a .sif file")
        if name is None:
            self.name = sif_file.split('/')[-1].split('.')[0]
            self.shot = self.name.split('_')[0]
        data, info = sif_parser.np_open(sif_file)
        self.data = data[0]
        print(f'Loaded Shot {self.shot}; {self.name}')
        if plot:
            plt.imshow(self.data,cmap='inferno')
            plt.show()

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

    def remove_bins(self, fibers):
        if isinstance(fibers, int):
            fibers = [fibers]
        self.bins = [b for b in self.bins if b.fiber not in fibers]
        print(f"Removed fiber(s) {fibers}. Remaining: {[b.fiber for b in self.bins]}")

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

if __name__ == "__main__":
    main()
