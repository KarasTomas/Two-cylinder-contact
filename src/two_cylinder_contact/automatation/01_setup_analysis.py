"""Tools for setting the analysis environment for the two-cylinder contact problem."""

import csv
import os
from typing import Any, List, Tuple

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
    """Read gear_parameters.csv and extract available gear labels from column headers.

    Returns list of gear labels and validation status.
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

    # Read gear parameters to extract labels using pandas
    try:
        gear_params_df = pd.read_csv(gear_params_file)
        measured_data_df = pd.read_csv(measured_data_file)

        gear_labels = [
            col for col in gear_params_df.columns if col.lower() != "parameter"
        ]
        measured_labels = [
            col for col in measured_data_df.columns if col.lower() != "parameter"
        ]

        print("\n--- Gear Labels Analysis ---")
        print(
            f"📊 Found {len(gear_labels)} geometries in gear_parameters.csv: {gear_labels}"
        )
        print(
            f"📊 Found {len(measured_labels)} geometries in measured_deformation.csv: {measured_labels}"
        )

        # Check for matching labels
        matching_labels = set(gear_labels) & set(measured_labels)
        missing_in_measured = set(gear_labels) - set(measured_labels)
        missing_in_params = set(measured_labels) - set(gear_labels)

        if matching_labels:
            print(f"✅ Matching geometries: {sorted(matching_labels)}")
        if missing_in_measured:
            print(
                f"⚠️  Missing in measured_deformation.csv: {sorted(missing_in_measured)}"
            )
        if missing_in_params:
            print(f"⚠️  Missing in gear_parameters.csv: {sorted(missing_in_params)}")

        return sorted(matching_labels), True

    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return [], False


def create_abaqus_config(gear_labels: List[str]) -> bool:
    """Create abaqus_config.csv with configuration for all gear geometries.

    Format: parameter,gear_label1,gear_label2,...
    """
    config_file = "processed_data/abaqus_config.csv"

    print("\n--- Creating Configuration ---")

    # Default configuration parameters
    default_config = {
        "calculate": True,
        "elastic_modulus": 3200,
        "poisson_ratio": 0.4,
        "arc_length": 6,
        "available_load_types": "F,D",
        "selected_load_type": "D",
    }

    # Create config data structure
    config_data: List[List[Any]] = []

    # Add default parameters (same for all geometries)
    for param_name, default_value in default_config.items():
        row = [param_name] + [default_value] * len(gear_labels)
        config_data.append(row)

    # Add model_name_prefix (unique for each geometry)
    model_prefix_row = ["model_name_prefix"] + [f"{label}" for label in gear_labels]
    config_data.append(model_prefix_row)

    # Write to CSV
    try:
        with open(config_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            header = ["parameter"] + gear_labels
            writer.writerow(header)

            # Write config data
            writer.writerows(config_data)

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

    try:
        # Read both files
        gear_params_df = pd.read_csv(gear_params_file)
        measured_data_df = pd.read_csv(measured_data_file)

        # Check for required parameters in gear_parameters.csv
        available_gear_params = (
            gear_params_df["parameter"].tolist()
            if "parameter" in gear_params_df.columns
            else []
        )

        # Read template to get expected parameters
        template_file = "initial_conditions/gear_parameters_template.csv"
        expected_params: List[str] = []
        if os.path.exists(template_file):
            try:
                template_df = pd.read_csv(template_file)
                expected_params = (
                    template_df["parameter"].tolist()
                    if "parameter" in template_df.columns
                    else []
                )
            except Exception:
                pass

        print(
            f"📋 Available parameters: {len(available_gear_params)} / {len(expected_params)} from template"
        )

        # Check data completeness for each geometry
        all_complete = True
        for label in gear_labels:
            issues = []

            # Check if geometry has data in both files
            if label not in gear_params_df.columns:
                issues.append("missing gear parameters")
            if label not in measured_data_df.columns:
                issues.append("missing measured data")

            # Check for empty values
            if label in gear_params_df.columns:
                empty_params = gear_params_df[label].isnull().sum()
                if empty_params > 0:
                    issues.append(f"{empty_params} empty parameter values")

            if label in measured_data_df.columns:
                empty_measured = measured_data_df[label].isnull().sum()
                if empty_measured > 0:
                    issues.append(f"{empty_measured} empty measured values")

            # Report status
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
