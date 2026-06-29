"""
Compute radial (v_r) and azimuthal (v_theta) flow velocity from two
90-degree Thomson scattering measurements (north and south collection
optics) viewing the same radially-propagating laser.
Geometry:
    Laser propagates along +r.
    North and South collection optics sit at +/-90 deg scattering angle,
    on opposite azimuthal sides of the laser path.
    For 90-deg scattering, each detector's LOS (Doppler) velocity is the
    projection of the flow velocity onto the bisector of (-laser dir)
    and (collection dir):
        v_LOS_north = (-v_r + v_theta) / sqrt(2)
        v_LOS_south = (-v_r - v_theta) / sqrt(2)
    Solving:
        v_r     = -(v_LOS_north + v_LOS_south) / sqrt(2)
        v_theta =  (v_LOS_north - v_LOS_south) / sqrt(2)
    (Assumes north's transverse direction is +theta, south's is -theta.
     Flip SIGN_CONVENTION below to -1 if this turns out to be backwards.)
Inputs: the "radius_vs_params.csv"-style files produced by the fiber
fitting script (one for north, one for south), each with a radius_um
column and an ion_speed_0 column (the LOS velocity used here).
Just edit the paths below and run.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------
# EDIT THESE
NORTH_CSV_PATH = r"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts-out/north"
SOUTH_CSV_PATH = r"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts-out/south"
OUTPUT_CSV_PATH = r"/Users/orenyang/Documents/UCSD_Lab/cornell_bz_paper/ICOPS_presentation/ts-out/o"
VELOCITY_COLUMN = "ion_speed_0"
# +1 : north transverse dir = +theta, south = -theta (default assumption)
# -1 : flip if it turns out to be backwards
SIGN_CONVENTION = 1
# ----------------------------------------------------------------------
SQRT2 = np.sqrt(2.0)


def plot_velocities_presentation(radii, v_r, v_theta):
    """Presentation-quality figure: v_r and v_theta vs radius, same axes."""
    plt.figure(figsize=(8, 6))

    plt.plot(
        radii, v_r/1000, "o-", color="steelblue",
        markersize=7, markerfacecolor="steelblue", markeredgewidth=1.5,
        linewidth=1.5, label=r"$v_r$",
    )
    plt.plot(
        radii, v_theta/1000, "s-", color="firebrick",
        markersize=7, markerfacecolor="firebrick", markeredgewidth=1.5,
        linewidth=1.5, label=r"$v_\theta$",
    )

    plt.axhline(0, color="gray", linewidth=1, linestyle="--", zorder=0)

    plt.xlabel("Radius (μm)", fontsize=16)
    plt.ylabel("Velocity (km/s)", fontsize=16)
    plt.title("Flow Velocity vs Radius", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=13)
    plt.legend(fontsize=13)
    ax = plt.gca()
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def main():
    north_df = pd.read_csv(NORTH_CSV_PATH)
    south_df = pd.read_csv(SOUTH_CSV_PATH)
    # Match rows by radius_um (same physical radius from pinch axis)
    merged = pd.merge(
        north_df[["radius_um", VELOCITY_COLUMN]].rename(
            columns={VELOCITY_COLUMN: "v_los_north"}
        ),
        south_df[["radius_um", VELOCITY_COLUMN]].rename(
            columns={VELOCITY_COLUMN: "v_los_south"}
        ),
        on="radius_um",
        how="inner",
    ).sort_values("radius_um").reset_index(drop=True)
    if merged.empty:
        print("No matching radius_um values found between north and south files.")
        return
    v_los_n = merged["v_los_north"].values
    v_los_s = merged["v_los_south"].values
    v_r = -(v_los_n + v_los_s) / SQRT2
    v_theta = SIGN_CONVENTION * (v_los_n - v_los_s) / SQRT2
    merged["v_r"] = v_r
    merged["v_theta"] = v_theta
    merged.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved v_r / v_theta table to: {OUTPUT_CSV_PATH}")
    print(merged)
    radii = merged["radius_um"].values

    plot_velocities_presentation(radii, v_r, v_theta)


if __name__ == "__main__":
    main()
