"""Tools for setting the analysis environment for the two-cylinder contact problem."""

import os
from typing import List, Tuple

import pandas as pd


def create_directory_structure() -> None:
    """Create the required directory structure if it doesn't exist."""
    directories = ["processed_data", "abaqus_work", "results"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")


def discover_gear_labels() -> Tuple[List[str], bool]:
    """Discover geometry IDs from gear_parameters.csv where calculate is True, and check if they have corresponding data in measured_deformation.csv.

    Returns:
        Tuple of (list of valid geometry IDs, bool indicating if files are OK)
    """
    gear_params_file = "initial_conditions/gear_parameters.csv"
    measured_data_file = "initial_conditions/measured_deformation.csv"

    print("\n--- Scanning for data files ---")

    # Check if required files exist
    files_status = {
        "gear_parameters.csv": os.path.exists(gear_params_file),
        "measured_deformation.csv": os.path.exists(measured_data_file),
    }

    for file, exists in files_status.items():
        status = "✅ Found" if exists else "❌ Missing"
        print(f"{status}: {file}")

    if not all(files_status.values()):
        print("❌ Required files missing! Cannot proceed.")
        return [], False

    try:
        gear_params_df = pd.read_csv(gear_params_file)
        measured_data_df = pd.read_csv(measured_data_file)

        # Get geometry_ids where calculate is True/1/yes
        calc_mask = (
            gear_params_df["calculate"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        gear_labels = gear_params_df.loc[calc_mask, "geometry_id"].astype(str).tolist()

        # Get geometry_ids present in measured_deformation.csv
        measured_labels = measured_data_df["geometry_id"].astype(str).unique().tolist()

        print("\n--- Gear Labels Analysis ---")
        print(
            f"📊 Found {len(gear_labels)} geometries to calculate in gear_parameters.csv: {gear_labels}"
        )
        print(
            f"📊 Found {len(measured_labels)} geometries in measured_deformation.csv: {measured_labels}"
        )

        # Check for matching labels
        matching_labels = sorted(set(gear_labels) & set(measured_labels))
        missing_in_measured = sorted(set(gear_labels) - set(measured_labels))

        if matching_labels:
            print(
                f"✅ Geometries with both parameters and measured data: {matching_labels}"
            )
        if missing_in_measured:
            print(f"⚠️  Missing measured data for: {missing_in_measured}")

        return matching_labels, True

    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return [], False


def create_abaqus_config(gear_labels: List[str]) -> bool:
    """Create abaqus_config.csv with one row per geometry, columns for all config parameters."""
    config_file = "processed_data/abaqus_config.csv"
    gear_params_file = "initial_conditions/gear_parameters.csv"

    print("\n--- Creating Configuration ---")

    # Default configuration parameters
    default_config = {
        "calculate": True,
        "arc_length": 6,
        "selected_load_type": "D",
    }

    try:
        gear_params_df = pd.read_csv(gear_params_file)
        config_df = gear_params_df[
            gear_params_df["geometry_id"].isin(gear_labels)
        ].copy()

        # Set/overwrite default config values
        for key, value in default_config.items():
            config_df[key] = value

        # Keep only relevant columns (add/remove as needed)
        columns = [
            "geometry_id",
            "calculate",
            "elastic_modulus",
            "poisson_ratio",
            "arc_length",
            "selected_load_type",
        ]
        config_df = config_df[columns]

        config_df.to_csv(config_file, index=False)
        print(f"✅ Created configuration file: {config_file}")
        return True

    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return False


def create_results_directories(gear_labels: List[str]) -> None:
    """Create results subdirectories for each geometry."""
    print("\n--- Creating Results Directories ---")

    for label in gear_labels:
        results_dir = f"results/{label}_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"✅ Created: {results_dir}")
        else:
            print(f"📁 Already exists: {results_dir}")


def validate_data_completeness(gear_labels: List[str]) -> bool:
    """Validate that all required data is available for each geometry."""
    print("\n--- Data Completeness Check ---")

    gear_params_file = "initial_conditions/gear_parameters.csv"
    measured_data_file = "initial_conditions/measured_deformation.csv"
    template_file = "initial_conditions/gear_parameters_template.csv"

    try:
        gear_params_df = pd.read_csv(gear_params_file)
        measured_data_df = pd.read_csv(measured_data_file)

        # Optionally load required columns from template
        required_columns = [
            "geometry_id",
            "center_distance",
            "pressure_angle",
            "normal_module",
            "face_width",
            "tooth_count_pinion",
            "tooth_count_gear",
            "profile_shift_coefficient_pinion",
            "profile_shift_coefficient_gear",
            "elastic_modulus",
            "poisson_ratio",
            "calculate",
        ]
        if os.path.exists(template_file):
            template_df = pd.read_csv(template_file)
            if "parameter" in template_df.columns:
                required_columns = template_df["parameter"].tolist()

        all_complete = True
        for label in gear_labels:
            issues = []

            # Check gear_parameters.csv row
            row = gear_params_df[gear_params_df["geometry_id"] == label]
            if row.empty:
                issues.append("missing in gear_parameters.csv")
            else:
                for col in required_columns:
                    if (
                        col not in row.columns
                        or pd.isnull(row.iloc[0][col])
                        or str(row.iloc[0][col]).strip() == ""
                    ):
                        issues.append(f"missing or empty: {col}")

            # Check measured_deformation.csv rows
            md_rows = measured_data_df[measured_data_df["geometry_id"] == label]
            if md_rows.empty:
                issues.append("missing in measured_deformation.csv")
            else:
                if (
                    md_rows["g_alpha"].isnull().any()
                    or md_rows["contact_width_S"].isnull().any()
                ):
                    issues.append("missing or empty g_alpha/contact_width_S")

            if issues:
                print(f"⚠️  {label}: {', '.join(issues)}")
                all_complete = False
            else:
                print(f"✅ {label}: Complete")

        return all_complete

    except Exception as e:
        print(f"❌ Error validating data: {e}")
        return False


def print_project_status(
    gear_labels: List[str], config_created: bool, data_complete: bool
) -> None:
    """Print overall project status and next steps."""
    print(f"\n{'=' * 50}")
    print("📊 PROJECT STATUS SUMMARY")
    print(f"{'=' * 50}")

    print(f"🎯 Geometries found: {len(gear_labels)}")
    if gear_labels:
        print(f"   Labels: {', '.join(gear_labels)}")

    print("📁 Directory structure: ✅ Complete")
    print(f"⚙️  Configuration file: {'✅ Created' if config_created else '❌ Failed'}")
    print(
        f"📋 Data completeness: {'✅ Complete' if data_complete else '⚠️  Issues found'}"
    )

    print("\n📋 NEXT STEPS:")
    if gear_labels and config_created:
        print("1. ✅ Review configuration in: processed_data/abaqus_config.csv")
        print("2. 🔧 Run 02_geometry_creation.py to generate additional geometry data")
        print("3. 📊 Create downsampled data for Abaqus analysis")
        print("4. 🔄 Run Abaqus analysis from abaqus_work/ directory")
    else:
        print("1. ❌ Fix data issues in initial_conditions/ folder")
        print("2. 🔄 Re-run this setup script")

    print(f"{'=' * 50}")


def main() -> None:
    """Main function to set up analysis environment."""
    print("=" * 50)
    print("🔧 TWO-CYLINDER CONTACT ANALYSIS SETUP")
    print("=" * 50)

    # Step 1: Create directory structure
    print("\nSTEP 1: Creating directory structure...")
    create_directory_structure()

    # Step 2: Discover geometries from gear_parameters.csv
    print("\nSTEP 2: Discovering gear geometries...")
    gear_labels, files_ok = discover_gear_labels()

    if not files_ok or not gear_labels:
        print("\n❌ Setup failed: Missing or invalid data files")
        return

    # Step 3: Create configuration file
    print("\nSTEP 3: Creating Abaqus configuration...")
    config_created = create_abaqus_config(gear_labels)

    # Step 4: Create results directories
    print("\nSTEP 4: Setting up results directories...")
    create_results_directories(gear_labels)

    # Step 5: Validate data completeness
    print("\nSTEP 5: Validating data completeness...")
    data_complete = validate_data_completeness(gear_labels)

    # Step 6: Print project status
    print_project_status(gear_labels, config_created, data_complete)


if __name__ == "__main__":
    main()
