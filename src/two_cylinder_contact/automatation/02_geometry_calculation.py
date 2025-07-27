"""Tools for calculating gear geometry parameters for the two-cylinder contact problem.

This module handles the calculation of gear contact mechanics parameters,
converting complex gear geometry into simplified two-cylinder contact model.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.geometry_utils import (
    calculate_addendum_lowering,
    calculate_contact_path_parameters,
    calculate_deformation_loading,
    calculate_equivalent_rho,
    calculate_load_delta,
    calculate_min_equivalent_radii,
    log_calculation_results,
    save_geometry_parameters,
    save_simulation_data,
)

TOOL_PROFILE = {
    "bottom_clearance": 0.25,
    "addendum_coefficient": 1.0,
    "dedendum_coefficient": 1.25,
    "fillet_radius_coefficient": 0.38,
}


def get_geometries_to_calculate() -> List[str]:
    """Get list of geometries that need calculation from gear_parameters.csv.

    Returns:
        List of gear labels that have calculate=True
    """
    gear_params_file = "initial_conditions/gear_parameters.csv"

    if not os.path.exists(gear_params_file):
        raise FileNotFoundError(f"Gear parameters file not found: {gear_params_file}")

    try:
        df = pd.read_csv(gear_params_file)

        # Only select rows where calculate is True/1/yes (case-insensitive)
        geometries_to_calc = df[
            df["calculate"].astype(str).str.lower().isin(["true", "1", "yes"])
        ]["geometry_id"].tolist()

        # Ensure the result is a list of str
        return [str(g) for g in geometries_to_calc]

    except Exception as e:
        raise RuntimeError(f"Error reading gear parameters file: {e}")


def load_gear_parameters(gear_label: str) -> Dict[str, float]:
    """Load gear parameters from gear_parameters.csv for a specific geometry.

    Args:
        gear_label: Label identifying the specific gear geometry

    Returns:
        Dictionary containing gear parameters
    """
    gear_params_file = "initial_conditions/gear_parameters.csv"

    if not os.path.exists(gear_params_file):
        raise FileNotFoundError(f"Gear parameters file not found: {gear_params_file}")

    try:
        df = pd.read_csv(gear_params_file)

        # Find the row for the requested gear_label
        row = df[df["geometry_id"] == gear_label]
        if row.empty:
            raise ValueError(f"Gear label '{gear_label}' not found in parameters file")

        # Convert row to dictionary (excluding geometry_id and calculate)
        params = row.iloc[0].to_dict()
        params.pop("geometry_id", None)
        params.pop("calculate", None)
        # Convert all values to float where possible
        for k, v in params.items():
            try:
                params[k] = float(v)
            except (ValueError, TypeError):
                pass  # Keep as is if not convertible

        return params

    except Exception as e:
        raise RuntimeError(f"Error loading gear parameters: {e}")


def validate_gear_parameters(params: Dict[str, float]) -> List[str]:
    """Validate that all required parameters are present and reasonable.

    Currently set to pass all validations for development.

    Args:
        params: Dictionary containing gear parameters

    Returns:
        List of validation error messages (empty if all OK)
    """
    # For now, always pass validation
    # TODO: Implement actual validation when needed
    return []


def calculate_gear_geometry(params: Dict[str, float]) -> Dict[str, float]:
    """Calculate gear geometry parameters for two-cylinder contact model.

    This function computes the key geometric parameters needed for gear contact analysis
    including pitch diameters, base diameters, addendum diameters, contact intervals,
    and equivalent radii for the two-cylinder model.

    Args:
        params: Dictionary containing input gear parameters including:
            - normal_module: Module of the gear [mm]
            - tooth_count_pinion: Number of teeth on pinion
            - tooth_count_gear: Number of teeth on gear
            - pressure_angle: Pressure angle [degrees]
            - profile_shift_coefficient_pinion: Profile shift coefficient for pinion
            - profile_shift_coefficient_gear: Profile shift coefficient for gear
            - center_distance: Center distance between gears [mm]

    Returns:
        Dictionary containing calculated geometry parameters with keys:
            - pitch_diameter_pinion/gear: Pitch diameters [mm]
            - base_diameter_pinion/gear: Base diameters [mm]
            - addendum_diameter_pinion/gear: Addendum diameters [mm]
            - equivalent_pinion/gear_radius: Equivalent radii for contact [mm]
            - And various intermediate calculations

    Raises:
        KeyError: If required parameters are missing from input
        ValueError: If calculated values are physically invalid
    """
    calculated_params: Dict[str, float] = {}

    try:
        # Extract basic parameters
        module = params["normal_module"]
        z1 = params["tooth_count_pinion"]
        z2 = params["tooth_count_gear"]
        pressure_angle_deg = params["pressure_angle"]
        x1 = params["profile_shift_coefficient_pinion"]
        x2 = params["profile_shift_coefficient_gear"]
        center_distance = params["center_distance"]

        # Convert pressure angle to radians
        pressure_angle_rad = np.deg2rad(pressure_angle_deg)

        # Calculate pitch diameters
        pitch_d_pinion = module * z1
        pitch_d_gear = module * z2
        calculated_params["pitch_diameter_pinion"] = pitch_d_pinion
        calculated_params["pitch_diameter_gear"] = pitch_d_gear

        # Calculate base diameters
        base_d_pinion = pitch_d_pinion * np.cos(pressure_angle_rad)
        base_d_gear = pitch_d_gear * np.cos(pressure_angle_rad)
        calculated_params["base_diameter_pinion"] = base_d_pinion
        calculated_params["base_diameter_gear"] = base_d_gear

        # Calculate addendum lowering
        addendum_lowering = calculate_addendum_lowering(
            pitch_d_pinion, x1, x2, module, center_distance
        )
        calculated_params["addendum_lowering"] = addendum_lowering

        # Calculate addendum diameters
        addendum_coeff = TOOL_PROFILE["addendum_coefficient"]
        addendum_d_pinion = pitch_d_pinion + 2 * module * (
            addendum_coeff + x1 - addendum_lowering
        )
        addendum_d_gear = pitch_d_gear + 2 * module * (
            addendum_coeff + x2 - addendum_lowering
        )
        calculated_params["addendum_diameter_pinion"] = addendum_d_pinion
        calculated_params["addendum_diameter_gear"] = addendum_d_gear

        # Calculate rolling pressure angle
        rolling_pressure_angle = np.arccos(base_d_pinion / center_distance)
        calculated_params["rolling_pressure_angle"] = rolling_pressure_angle

        # Calculate contact path parameters
        contact_params = calculate_contact_path_parameters(
            base_d_pinion,
            addendum_d_pinion,
            addendum_d_gear,
            center_distance,
            module,
            pressure_angle_rad,
        )
        calculated_params.update(contact_params)

        # Calculate equivalent radii for two-cylinder model
        equivalent_radii = calculate_min_equivalent_radii(base_d_pinion, contact_params)
        calculated_params.update(equivalent_radii)

        # Log calculation results
        log_calculation_results(calculated_params)

        return calculated_params

    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e}")
    except (ValueError, np.linalg.LinAlgError) as e:
        raise ValueError(f"Invalid calculation result: {e}")


def load_geometry_parameters(gear_label: str) -> Dict[str, float]:
    """Load geometry parameters for a specific gear label from saved file."""
    geometry_file = f"processed_data/{gear_label}_geometry_parameters.csv"
    try:
        df = pd.read_csv(geometry_file)
        return {
            k: float(v) for k, v in df.set_index("parameter")["value"].to_dict().items()
        }
    except Exception as e:
        raise OSError(f"Failed to load geometry parameters from {geometry_file}: {e}")


def calculate_simulation_data(
    gear_label: str, points: int = 1000
) -> Dict[str, List[float]]:
    """Calculate and return simulation data for a specific gear label."""
    geometry_params = load_geometry_parameters(gear_label)
    contact_path = np.linspace(0, geometry_params["g_alpha"], num=points).tolist()
    N1E = geometry_params["N_1E"]
    N2F = geometry_params["N_2F"]
    calc_sim_data = {}
    eq_rho = calculate_equivalent_rho(
        (
            geometry_params["base_diameter_pinion"] / 2,
            geometry_params["base_diameter_gear"] / 2,
        ),
        (N1E, N2F),
        contact_path,
    )
    deformation_loading = calculate_deformation_loading(eq_rho, gear_label)
    calc_sim_data["contact_path"] = contact_path
    calc_sim_data["rho_pinion"] = eq_rho["rho_pinion"]
    calc_sim_data["R_y_pinion"] = eq_rho["R_y_pinion"]
    calc_sim_data["rho_gear"] = eq_rho["rho_gear"]
    calc_sim_data["R_y_gear"] = eq_rho["R_y_gear"]
    calc_sim_data["deformation_loading"] = deformation_loading["deformation_loading"]
    load_delta = calculate_load_delta(calc_sim_data)
    calc_sim_data["load_delta"] = load_delta

    return calc_sim_data


def calculate_geometry_for_label(gear_label: str) -> bool:
    """Calculate and save geometry data for a single gear label."""
    try:
        print(f"\n--- Calculating geometry for {gear_label} ---")
        params = load_gear_parameters(gear_label)
        errors = validate_gear_parameters(params)
        if errors:
            print("❌ Parameter validation failed:")
            for error in errors:
                print(f"   {error}")
            return False
        calculated_params = calculate_gear_geometry(params)
        save_geometry_parameters(gear_label, params, calculated_params)
        print(f"✅ Geometry calculated and saved for {gear_label}")
        return True
    except Exception as e:
        print(f"❌ Error calculating geometry for {gear_label}: {e}")
        return False


def main() -> None:
    print("=" * 60)
    print("🧮 TWO-CYLINDER CONTACT GEOMETRY CALCULATION")
    print("=" * 60)
    try:
        print("\nSTEP 2: Loading geometries to calculate...")
        geometries_to_calc = get_geometries_to_calculate()
        if not geometries_to_calc:
            print("❌ No geometries marked for calculation in config file")
            return
        print(f"📋 Found {len(geometries_to_calc)} geometries: {geometries_to_calc}")
        print("\nSTEP 3: Processing geometries...")
        successful_calculations = 0
        for gear_label in geometries_to_calc:
            if calculate_geometry_for_label(gear_label):
                successful_calculations += 1

        print("\nSTEP 4: Calculating and saving simulation data...")
        for gear_label in geometries_to_calc:
            try:
                sim_data = calculate_simulation_data(gear_label, 1000)
                save_simulation_data(sim_data, gear_label)
                print(f"✅ Simulation data calculated and saved for {gear_label}")
            except Exception as e:
                print(
                    f"❌ Error calculating/saving simulation data for {gear_label}: {e}"
                )

        print(f"\n{'=' * 60}")
        print("📊 CALCULATION SUMMARY")
        print(f"{'=' * 60}")
        print(
            f"✅ Successful: {successful_calculations}/{len(geometries_to_calc)} geometries"
        )
        if successful_calculations == len(geometries_to_calc):
            print("🎉 All geometry calculations completed successfully!")
        else:
            print("⚠️  Some calculations failed. Check error messages above.")
    except Exception as e:
        print(f"❌ Error in geometry calculation: {e}")


def read_txt_file(filepath: str) -> List[float]:
    """Read a tab-separated file, skipping comment lines, and return a list of floats."""
    with open(filepath, "r") as f:
        lines = [line for line in f if not line.strip().startswith("//")]
    # Join all lines and split by tab
    data = "\t".join([line.strip() for line in lines]).split("\t")
    return [float(x) for x in data if x.strip()]


if __name__ == "__main__":
    main()
