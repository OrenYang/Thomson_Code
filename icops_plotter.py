"""
Thomson scattering fit analysis
--------------------------------
For each fiber (0-18), scans all available fit attempts (num 1-5),
computes R^2 for each from the *_fit_spectrum.csv file (data vs best_fit),
picks the best (highest R^2) fit per fiber, then plots every fitted
parameter (from the corresponding fiber<N>_<num>.csv file) vs fiber
number, each parameter on its own figure.

Expected files in FOLDER_PATH, for each fiber N and attempt num:
    fiber<N>_<num>.csv               -> columns: parameter,value,stderr
    fiber<N>_<num>_fit_spectrum.csv  -> columns: wavelength_nm,data,best_fit

Just edit FOLDER_PATH below and run.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# PRESENTATION FIGURE: T_e vs radius with error bars (mean +/- std across
# all fits per fiber). Call plot_Te_presentation(radii, means, stds) once
# those are computed further down - everything else in the file is
# unchanged.
# ----------------------------------------------------------------------
def plot_Te_presentation(radii, te_mean, te_std, ti0_mean=None, ti0_std=None,
                          ti1_mean=None, ti1_std=None):
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        radii, te_mean, yerr=te_std,
        fmt="o-", color="steelblue", ecolor="steelblue",
        markersize=7, markerfacecolor="steelblue", markeredgewidth=1.5,
        elinewidth=1.5, capsize=4, capthick=1.5, label=r"$T_e$",
    )
    if ti0_mean is not None:
        plt.errorbar(
            radii, ti0_mean, yerr=ti0_std,
            fmt="s-", color="firebrick", ecolor="firebrick",
            markersize=7, markerfacecolor="firebrick", markeredgewidth=1.5,
            elinewidth=1.5, capsize=4, capthick=1.5, label=r"$T_i$ (liner)",
        )
    if ti1_mean is not None:
        plt.errorbar(
            radii, ti1_mean, yerr=ti1_std,
            fmt="^-", color="seagreen", ecolor="seagreen",
            markersize=7, markerfacecolor="seagreen", markeredgewidth=1.5,
            elinewidth=1.5, capsize=4, capthick=1.5, label=r"$T_i$ (target)",
        )
    plt.xlabel("Radius (μm)", fontsize=16)
    plt.ylabel("Temperature (eV)", fontsize=16)
    plt.title("Temperature vs Radius", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=13)
    plt.legend(fontsize=13)
    ax = plt.gca()
    ax.grid(False)
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# EDIT THIS: path to the folder containing all your fiber csv files
FOLDER_PATH = r"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts-out/7237s"

# EDIT THIS: path/filename for the output CSV of radius vs parameters
OUTPUT_CSV_PATH = r"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts-out/out6"

# EDIT THIS: path/filename for the averaged (mean +/- std across all fits) CSV
AVERAGED_OUTPUT_CSV_PATH = r"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts-out/out6_averaged.csv"

# EDIT THIS: which fiber number corresponds to r = 0
ZERO_FIBER = 8

# EDIT THIS: spacing between adjacent fibers, in micrometers
SPACING_UM = 250.0

# EDIT THIS: +1 if increasing fiber number = increasing (positive) radius,
#            -1 if increasing fiber number = decreasing (negative) radius
DIRECTION = 1
# ----------------------------------------------------------------------

FIBER_RANGE = range(0, 19)   # fibers 0-18
NUM_RANGE = range(1, 6)      # nums 1-5


def fiber_to_radius(fiber):
    """Convert a fiber number to a radius in um, given ZERO_FIBER, SPACING_UM,
    and DIRECTION."""
    return DIRECTION * (fiber - ZERO_FIBER) * SPACING_UM


def compute_r_squared(data, best_fit):
    """Standard R^2 = 1 - SS_res/SS_tot"""
    data = np.asarray(data, dtype=float)
    best_fit = np.asarray(best_fit, dtype=float)
    mask = ~(np.isnan(data) | np.isnan(best_fit))
    data = data[mask]
    best_fit = best_fit[mask]
    if len(data) == 0:
        return np.nan
    ss_res = np.sum((data - best_fit) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def find_best_fit_for_fiber(folder, fiber):
    """Return (best_num, best_r2, params_dict) for a given fiber, or None if
    no valid fit files were found."""
    best_num = None
    best_r2 = -np.inf
    for num in NUM_RANGE:
        spectrum_path = os.path.join(folder, f"fiber{fiber}_{num}_fit_spectrum.csv")
        params_path = os.path.join(folder, f"fiber{fiber}_{num}.csv")

        if not (os.path.isfile(spectrum_path) and os.path.isfile(params_path)):
            continue

        try:
            spec_df = pd.read_csv(spectrum_path)
            r2 = compute_r_squared(spec_df["data"], spec_df["best_fit"])
        except Exception as e:
            print(f"  [warn] could not read/compute R^2 for {spectrum_path}: {e}")
            continue

        if np.isnan(r2):
            continue

        if r2 > best_r2:
            best_r2 = r2
            best_num = num

    if best_num is None:
        return None

    # Load the parameters of the winning fit
    params_path = os.path.join(folder, f"fiber{fiber}_{best_num}.csv")
    params_df = pd.read_csv(params_path)
    params_dict = dict(zip(params_df["parameter"], params_df["value"]))

    return best_num, best_r2, params_dict


def collect_all_fits_for_fiber(folder, fiber):
    """Return a list of params_dict, one per valid (num) fit found for this
    fiber (no best-fit selection - just every fit that has both files)."""
    all_params = []
    for num in NUM_RANGE:
        spectrum_path = os.path.join(folder, f"fiber{fiber}_{num}_fit_spectrum.csv")
        params_path = os.path.join(folder, f"fiber{fiber}_{num}.csv")

        if not (os.path.isfile(spectrum_path) and os.path.isfile(params_path)):
            continue

        try:
            params_df = pd.read_csv(params_path)
            params_dict = dict(zip(params_df["parameter"], params_df["value"]))
            all_params.append(params_dict)
        except Exception as e:
            print(f"  [warn] could not read {params_path}: {e}")
            continue

    return all_params


def plot_params_with_error_bars(fibers_sorted, radii, all_fits_by_fiber, output_csv_path=None):
    """For each fiber, average every available fit's parameters and use the
    standard deviation across fits as the error bar. Plots each parameter
    vs radius with error bars, and optionally saves a CSV."""

    # Union of all parameter names seen across all fibers/fits
    all_param_names = []
    for fits in all_fits_by_fiber.values():
        for params_dict in fits:
            for p in params_dict.keys():
                if p not in all_param_names:
                    all_param_names.append(p)

    means = {param: [] for param in all_param_names}
    stds = {param: [] for param in all_param_names}
    n_fits_used = []

    for fiber in fibers_sorted:
        fits = all_fits_by_fiber.get(fiber, [])
        n_fits_used.append(len(fits))
        for param in all_param_names:
            values = [d[param] for d in fits if param in d and not np.isnan(d[param])]
            if len(values) == 0:
                means[param].append(np.nan)
                stds[param].append(np.nan)
            else:
                means[param].append(np.mean(values))
                stds[param].append(np.std(values, ddof=1) if len(values) > 1 else 0.0)

    if output_csv_path is not None:
        table = {"fiber": fibers_sorted, "radius_um": radii, "n_fits_used": n_fits_used}
        for param in all_param_names:
            table[f"{param}_mean"] = means[param]
            table[f"{param}_std"] = stds[param]
        pd.DataFrame(table).to_csv(output_csv_path, index=False)
        print(f"Saved averaged parameters (with std across fits) to: {output_csv_path}")

    # Presentation-quality T_e figure (see top of file)
    if "T_e_0" in means:
        plot_Te_presentation(radii, means["T_e_0"], stds["T_e_0"], means["T_i_0"], stds["T_i_0"], means["T_i_1"], stds["T_i_1"])

    for param in all_param_names:
        plt.figure()
        plt.errorbar(radii, means[param], yerr=stds[param], fmt="o-", capsize=3)
        plt.xlabel("Radius (um)")
        plt.ylabel(param)
        plt.title(f"{param} vs Radius (mean +/- std across fits)")
        plt.grid(True)

    plt.show()



def plot_lineouts_with_fit(folder, results, offset_step=None):
    """Recreate the stacked 'Lineouts' plot, one color per fiber."""

    fibers_sorted = sorted(results.keys())

    spectra = {}
    for fiber in fibers_sorted:
        num = results[fiber]["num"]
        spectrum_path = os.path.join(
            folder,
            f"fiber{fiber}_{num}_fit_spectrum.csv"
        )

        if not os.path.isfile(spectrum_path):
            continue

        spec_df = pd.read_csv(spectrum_path)
        spectra[fiber] = spec_df

    if not spectra:
        print("No fit_spectrum files found for plotting lineouts.")
        return

    if offset_step is None:
        max_amp = max(
            df["data"].max() - df["data"].min()
            for df in spectra.values()
        )
        offset_step = max_amp * 1.1 if max_amp > 0 else 1.0

    plt.figure(figsize=(7, 9))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    n_fibers = len(fibers_sorted)
    for i, fiber in enumerate(fibers_sorted):
        if fiber not in spectra:
            continue

        spec_df = spectra[fiber]
        color = color_cycle[i % len(color_cycle)]
        offset = (n_fibers - 1 - i) * offset_step

        plt.plot(
            spec_df["wavelength_nm"],
            spec_df["data"] + offset,
            color=color,
            linewidth=1,
            alpha=0.8,
            label=f"F{fiber}"
        )

        plt.plot(
            spec_df["wavelength_nm"],
            spec_df["best_fit"] + offset,
            color="black",
            linestyle="--",
            linewidth=1
        )

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity (offset per fiber)")
    plt.title("Lineouts with best fit overlaid")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    results = {}  # fiber -> dict with 'num', 'r2', 'params'

    for fiber in FIBER_RANGE:
        result = find_best_fit_for_fiber(FOLDER_PATH, fiber)
        if result is None:
            print(f"Fiber {fiber}: no valid fit files found, skipping.")
            continue
        best_num, best_r2, params_dict = result
        results[fiber] = {"num": best_num, "r2": best_r2, "params": params_dict}
        print(f"Fiber {fiber}: best fit = num {best_num}, R^2 = {best_r2:.5f}")

    if not results:
        print("No fits found at all. Check FOLDER_PATH and file naming.")
        return

    fibers_sorted = sorted(results.keys())
    radii = [fiber_to_radius(f) for f in fibers_sorted]

    # Collect the union of all parameter names across all fibers
    all_param_names = []
    for f in fibers_sorted:
        for p in results[f]["params"].keys():
            if p not in all_param_names:
                all_param_names.append(p)

    # --- Build a table: fiber, radius_um, num_used, r2, then each parameter ---
    table = {
        "fiber": fibers_sorted,
        "radius_um": radii,
        "num_used": [results[f]["num"] for f in fibers_sorted],
        "r2": [results[f]["r2"] for f in fibers_sorted],
    }
    for param in all_param_names:
        table[param] = [results[f]["params"].get(param, np.nan) for f in fibers_sorted]

    '''out_df = pd.DataFrame(table)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved radius vs parameter table to: {OUTPUT_CSV_PATH}")

    # --- Plot R^2 of the best fit vs radius ---
    plt.figure()
    plt.plot(radii, table["r2"], "o-")
    plt.xlabel("Radius (um)")
    plt.ylabel("Best fit R$^2$")
    plt.title("Best-fit R$^2$ vs Radius")
    plt.grid(True)

    # --- Plot every parameter vs radius, each on its own figure ---
    for param in all_param_names:
        plt.figure()
        plt.plot(radii, table[param], "o-")
        plt.xlabel("Radius (um)")
        plt.ylabel(param)
        plt.title(f"{param} vs Radius (best fit)")
        plt.grid(True)

    plt.show()

    # --- Average every fit (not just the best) per fiber, with error bars ---
    all_fits_by_fiber = {fiber: collect_all_fits_for_fiber(FOLDER_PATH, fiber)
                          for fiber in fibers_sorted}
    plot_params_with_error_bars(fibers_sorted, radii, all_fits_by_fiber,
                                 output_csv_path=AVERAGED_OUTPUT_CSV_PATH)'''

    # --- "Lineouts" style plot: stacked spectra with best fit overlaid ---
    plot_lineouts_with_fit(FOLDER_PATH, results)


if __name__ == "__main__":
    main()
