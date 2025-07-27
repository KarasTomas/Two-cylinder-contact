# -*- coding: utf-8 -*-
"""Tool for selecting and tuning samples for two-cylinder contact simulation.

This script reads pre-calculated simulation data, applies a feature-based
sampling strategy to select the most representative points, and generates a
downsampled CSV file for Abaqus analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# USER CONFIGURATION - MODIFY THESE PARAMETERS
# ==============================================================================

# 1. Geometry to tune (must match a file in 'processed_data/{GEOMETRY_SUFFIX}_simulation_data.csv')
GEOMETRY_SUFFIX = "1805"

# 2. Sampling Parameters
TARGET_POINTS = 50  # The desired number of final sample points.
TRIM_START = 120  # Number of points to trim from the beginning of the data.
TRIM_END = 120  # Number of points to trim from the end of the data.

# 3. Feature-Based Sampler Configuration
# This strategy identifies distinct regions (plateau, slopes, knees) of the curve.
PLATEAU_THRESHOLD = 0.8  # Linear peak - Defines the vertical threshold (98% of max) for finding the peak plateau.
FLAT_SLOPE_TOLERANCE = 0.2  # Linear slope - Defines a "flat" slope for plateau detection (relative to max slope).

# Point allocation ratios (should sum roughly to 1.0)
PLATEAU_POINTS_RATIO = 0.25  # 25% of points on the peak plateau.
MAIN_SLOPE_POINTS_RATIO = 0.45  # 45% of points on the steepest up/down slopes.
SHALLOW_SLOPE_POINTS_RATIO = 0.30  # 30% of points on the shallow regions.

# 4. Final Output Control
OUTPUT_TO_CSV = False

# ==============================================================================


def load_simulation_data(geometry_label: str) -> pd.DataFrame:
    """Loads simulation data for a given geometry."""
    file_path = f"processed_data/gear_{geometry_label}_simulation_data.csv"
    if not os.path.exists(file_path):
        print(f"❌ Error: Data file not found at '{file_path}'")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded simulation data for '{geometry_label}': {len(df)} points")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()


def generate_feature_based_samples(data: pd.DataFrame, config: dict) -> dict:
    """Generates sample points using a feature-detection strategy."""
    load_data = data["load_delta"].values
    n_points = len(load_data)

    # -- Step 1: Calculate Derivatives --
    gradient = np.gradient(load_data)
    max_abs_gradient = np.abs(gradient).max()

    # -- Step 2: Detect Peak Plateau --
    peak_value = load_data.max()
    above_threshold_indices = np.where(
        load_data >= peak_value * config["plateau_threshold"]
    )[0]

    flat_indices = np.where(
        np.abs(gradient) < max_abs_gradient * config["flat_slope_tolerance"]
    )[0]

    plateau_indices = np.intersect1d(above_threshold_indices, flat_indices)
    plateau_start = plateau_indices.min() if len(plateau_indices) > 0 else -1
    plateau_end = plateau_indices.max() if len(plateau_indices) > 0 else -1

    if plateau_start == -1 or plateau_start == plateau_end:  # Fallback for sharp peaks
        plateau_start = plateau_end = np.argmax(load_data)

    print(
        f"Feature Detection: Peak Plateau found from index {plateau_start} to {plateau_end}"
    )

    # -- Step 3: Detect Main Slopes --
    pre_plateau_grad = gradient[:plateau_start]
    post_plateau_grad = gradient[plateau_end:]

    main_rise_center = np.argmax(pre_plateau_grad) if len(pre_plateau_grad) > 0 else -1
    main_fall_center = (
        plateau_end + np.argmin(post_plateau_grad) if len(post_plateau_grad) > 0 else -1
    )

    print(
        f"Feature Detection: Main Rise centered at {main_rise_center}, Main Fall at {main_fall_center}"
    )

    # -- Step 4: Allocate and Select Points --
    indices = set()

    # Allocate points to plateau
    num_plateau_pts = int(config["target_points"] * config["plateau_points_ratio"])
    if plateau_end > plateau_start:
        indices.update(
            np.linspace(plateau_start, plateau_end, num_plateau_pts, dtype=int)
        )
    else:
        indices.add(plateau_start)

    # Allocate points to main slopes
    num_main_slope_pts = int(
        config["target_points"] * config["main_slope_points_ratio"]
    )
    num_rise_pts = num_main_slope_pts // 2
    num_fall_pts = num_main_slope_pts - num_rise_pts

    # Define window for main slopes
    rise_window = n_points // 10
    fall_window = n_points // 10

    if main_rise_center != -1:
        rise_start = max(0, main_rise_center - rise_window)
        rise_end = min(plateau_start, main_rise_center + rise_window)
        indices.update(np.linspace(rise_start, rise_end, num_rise_pts, dtype=int))

    if main_fall_center != -1:
        fall_start = max(plateau_end, main_fall_center - fall_window)
        fall_end = min(n_points - 1, main_fall_center + fall_window)
        indices.update(np.linspace(fall_start, fall_end, num_fall_pts, dtype=int))

    # Allocate points to shallow slopes
    num_shallow_pts = config["target_points"] - len(indices)
    num_shallow_rise = num_shallow_pts // 2
    num_shallow_fall = num_shallow_pts - num_shallow_rise

    shallow_rise_end = max(0, main_rise_center - rise_window)
    if shallow_rise_end > 0:
        indices.update(np.linspace(0, shallow_rise_end, num_shallow_rise, dtype=int))

    shallow_fall_start = min(n_points - 1, main_fall_center + fall_window)
    if shallow_fall_start < n_points - 1:
        indices.update(
            np.linspace(shallow_fall_start, n_points - 1, num_shallow_fall, dtype=int)
        )

    final_indices = np.sort(list(indices))

    print(f"Sampling complete: Generated {len(final_indices)} points.")

    return {
        "indices": final_indices,
        "plateau_indices": np.arange(plateau_start, plateau_end + 1),
        "main_rise_center": main_rise_center,
        "main_fall_center": main_fall_center,
    }


def plot_sampling_results(
    original_data: pd.DataFrame,
    sampled_data: pd.DataFrame,
    results: dict,
    geometry_suffix: str,
) -> None:
    """Creates an enhanced visualization of the feature-based sampling."""
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 12), constrained_layout=True, sharex=True
    )
    fig.suptitle(
        f"Feature-Based Sampling for: {geometry_suffix} ({len(sampled_data)} points)",
        fontsize=18,
    )

    # --- Top Plot: Load Delta ---
    ax1 = axes[0]
    ax1.plot(
        original_data.index,
        original_data["load_delta"],
        "k-",
        alpha=0.2,
        label="Original Load Delta",
    )

    plateau_indices = results.get("plateau_indices", [])
    if len(plateau_indices) > 0:
        ax1.fill_between(
            original_data.index,
            original_data["load_delta"],
            where=np.isin(original_data.index, plateau_indices),
            color="red",
            alpha=0.3,
            label="Detected Plateau",
        )

    if results.get("main_rise_center", -1) != -1:
        ax1.axvline(
            x=results["main_rise_center"],
            color="blue",
            linestyle="--",
            label="Main Rise Center",
        )
    if results.get("main_fall_center", -1) != -1:
        ax1.axvline(
            x=results["main_fall_center"],
            color="green",
            linestyle="--",
            label="Main Fall Center",
        )

    ax1.plot(
        sampled_data.index,
        sampled_data["load_delta"],
        "ro",
        markersize=6,
        label="Sampled Points",
    )
    ax1.set_title("Load Delta with Detected Features")
    ax1.set_ylabel("Load Delta")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Bottom Plot: Radii ---
    ax2 = axes[1]
    ax2.plot(
        original_data.index,
        original_data["rho_pinion"],
        "b-",
        alpha=0.3,
        label="Original Pinion Radius",
    )
    ax2.plot(
        original_data.index,
        original_data["rho_gear"],
        "g-",
        alpha=0.3,
        label="Original Gear Radius",
    )
    ax2.plot(
        sampled_data.index,
        sampled_data["rho_pinion"],
        "bo",
        markersize=5,
        label="Sampled Pinion",
    )
    ax2.plot(
        sampled_data.index,
        sampled_data["rho_gear"],
        "go",
        markersize=5,
        label="Sampled Gear",
    )
    ax2.set_title("Equivalent Radii Sampling")
    ax2.set_ylabel("Radius (mm)")
    ax2.set_xlabel("Data Point Index")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.show()


def save_to_csv(sampled_data: pd.DataFrame, geometry_suffix: str) -> None:
    """Saves the downsampled data to the required CSV format."""
    output_filename = f"processed_data/Gears_{geometry_suffix}_downsampled_data.csv"
    output_df = pd.DataFrame(
        {
            "pinion_radius": sampled_data["rho_pinion"],
            "gear_radius": sampled_data["rho_gear"],
            "displacement_load": sampled_data["load_delta"],
            "g_alfa": sampled_data["contact_path"],
        }
    )
    output_df.to_csv(output_filename, index=False)
    print(f"\n✅ Success! CSV file saved: {output_filename} ({len(output_df)} points)")


def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("      Gear Analysis - Feature-Based Sampling Tuner")
    print("=" * 60)

    config = {
        "geometry_suffix": GEOMETRY_SUFFIX,
        "trim_start": TRIM_START,
        "trim_end": TRIM_END,
        "target_points": TARGET_POINTS,
        "plateau_threshold": PLATEAU_THRESHOLD,
        "flat_slope_tolerance": FLAT_SLOPE_TOLERANCE,
        "plateau_points_ratio": PLATEAU_POINTS_RATIO,
        "main_slope_points_ratio": MAIN_SLOPE_POINTS_RATIO,
        "shallow_slope_points_ratio": SHALLOW_SLOPE_POINTS_RATIO,
        "output_to_csv": OUTPUT_TO_CSV,
    }
    print(f"▶️  Tuning parameters for geometry: {config['geometry_suffix']}\n")

    original_data = load_simulation_data(config["geometry_suffix"])
    if original_data.empty:
        return

    trim_end = -config["trim_end"] if config["trim_end"] > 0 else None
    trimmed_data = original_data.iloc[config["trim_start"] : trim_end].reset_index(
        drop=True
    )
    print(f"Data points after trimming: {len(trimmed_data)}\n")

    print("Applying feature-based sampling strategy...")
    sampling_results = generate_feature_based_samples(trimmed_data, config)
    sampled_data = trimmed_data.iloc[sampling_results["indices"]]

    if not config["output_to_csv"]:
        print("\n" + "=" * 50)
        print("PREVIEW MODE")
        print(" -> Close the plot window to continue.")
        print(" -> To save, set OUTPUT_TO_CSV = True and re-run.")
        print("=" * 50)

    plot_sampling_results(
        trimmed_data, sampled_data, sampling_results, config["geometry_suffix"]
    )

    if config["output_to_csv"]:
        save_to_csv(sampled_data, config["geometry_suffix"])
        print("\nNEXT STEPS:")
        print("1. Process another geometry by changing GEOMETRY_SUFFIX.")
        print("2. Once all are sampled, proceed to the Abaqus analysis.")


if __name__ == "__main__":
    main()
