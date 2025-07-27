"""Tools for post-processing the results of the two-cylinder contact simulation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
GEOMETRY_SUFFIX = "179"
CSV_FILENAME = (
    "Gears_179_rP_23,02_rG_37,18_F_45,63_res.csv"  # Change to your actual file
)
# ==============================================================================


def main():
    """Simple script to test extracted CSV data"""
    # Path to CSV file
    csv_path = (
        Path("results")
        / f"Gears_{GEOMETRY_SUFFIX}_results"
        / "extracted_data"
        / CSV_FILENAME
    )

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        print("\nAvailable files:")
        data_dir = csv_path.parent
        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                print(f"  {file.name}")
        return

    # Read CSV data
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"✓ Data shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")

    # Create simple plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Extracted Data: {CSV_FILENAME}", fontsize=14)

    # Plot 1: U2 displacement
    ax1 = axes[0, 0]
    if "X_pinion_U2" in df.columns and "Y_pinion_U2" in df.columns:
        ax1.plot(
            df["X_pinion_U2"], df["Y_pinion_U2"], "b-", label="Pinion", linewidth=2
        )
    if "X_gear_U2" in df.columns and "Y_gear_U2" in df.columns:
        ax1.plot(df["X_gear_U2"], df["Y_gear_U2"], "r-", label="Gear", linewidth=2)
    ax1.set_title("U2 Displacement")
    ax1.set_xlabel("Distance along path")
    ax1.set_ylabel("U2 [mm]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Contact pressure
    ax2 = axes[0, 1]
    if "X_pinion_CPRESS" in df.columns and "Y_pinion_CPRESS" in df.columns:
        ax2.plot(
            df["X_pinion_CPRESS"],
            df["Y_pinion_CPRESS"],
            "b-",
            label="Pinion",
            linewidth=2,
        )
    if "X_gear_CPRESS" in df.columns and "Y_gear_CPRESS" in df.columns:
        ax2.plot(
            df["X_gear_CPRESS"], df["Y_gear_CPRESS"], "r-", label="Gear", linewidth=2
        )
    ax2.set_title("Contact Pressure")
    ax2.set_xlabel("Distance along path")
    ax2.set_ylabel("CPRESS [MPa]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Contact normal force
    ax3 = axes[1, 0]
    if "X_pinion_CFNORM" in df.columns and "Y_pinion_CFNORM" in df.columns:
        ax3.plot(
            df["X_pinion_CFNORM"],
            df["Y_pinion_CFNORM"],
            "b-",
            label="Pinion",
            linewidth=2,
        )
    if "X_gear_CFNORM" in df.columns and "Y_gear_CFNORM" in df.columns:
        ax3.plot(
            df["X_gear_CFNORM"], df["Y_gear_CFNORM"], "r-", label="Gear", linewidth=2
        )
    ax3.set_title("Contact Normal Force")
    ax3.set_xlabel("Distance along path")
    ax3.set_ylabel("CFNORM [N]")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Data summary
    ax4 = axes[1, 1]
    ax4.text(0.1, 0.8, f"File: {CSV_FILENAME}", transform=ax4.transAxes, fontsize=10)
    ax4.text(0.1, 0.7, f"Rows: {len(df)}", transform=ax4.transAxes, fontsize=10)
    ax4.text(
        0.1, 0.6, f"Columns: {len(df.columns)}", transform=ax4.transAxes, fontsize=10
    )

    # Show which variables were extracted
    variables = set()
    for col in df.columns:
        if "_" in col:
            var = col.split("_")[-1]  # Get variable name (last part)
            variables.add(var)

    ax4.text(
        0.1,
        0.5,
        f"Variables: {sorted(variables)}",
        transform=ax4.transAxes,
        fontsize=10,
    )
    ax4.text(
        0.1,
        0.3,
        f"Max U2: {df.filter(like='Y_').filter(like='U2').max().max():.4f}",
        transform=ax4.transAxes,
        fontsize=10,
    )
    ax4.text(
        0.1,
        0.2,
        f"Max CPRESS: {df.filter(like='Y_').filter(like='CPRESS').max().max():.2f}",
        transform=ax4.transAxes,
        fontsize=10,
    )
    ax4.set_title("Data Summary")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()

    # Print basic statistics
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    for col in df.columns:
        if col.startswith("Y_"):  # Only show Y (value) columns
            print(
                f"{col:20}: min={df[col].min():8.3f}, max={df[col].max():8.3f}, mean={df[col].mean():8.3f}"
            )


if __name__ == "__main__":
    main()
