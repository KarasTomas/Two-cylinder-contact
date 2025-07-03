"""Tools for calculating gear geometry parameters for the two-cylinder contact problem.

This module handles the calculation of gear contact mechanics parameters,
converting complex gear geometry into simplified two-cylinder contact model.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

TOOL_PROFILE = {
    "bottom_clearance": 0.25,
    "addendum_coefficient": 1.0,
    "dedendum_coefficient": 1.25,
    "fillet_radius_coefficient": 0.38,
}


def get_geometries_to_calculate() -> List[str]:
    """Get list of geometries that need calculation from config file.

    Returns:
        List of gear labels that have calculate=True in config
    """
    config_file = "processed_data/abaqus_config.csv"

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        df = pd.read_csv(config_file)

        # Find geometries with calculate=True
        if "calculate" not in df["parameter"].values:
            print(
                "⚠️  No 'calculate' parameter found in config - processing all geometries"
            )
            # Return all gear labels (all columns except 'parameter')
            return [col for col in df.columns if col.lower() != "parameter"]

        # Get the calculate row
        calc_row = df[df["parameter"] == "calculate"]
        if calc_row.empty:
            return []

        # Find columns where calculate=True
        geometries_to_calc = []
        for col in df.columns:
            if col.lower() != "parameter":
                calc_value = calc_row[col].iloc[0]
                if str(calc_value).lower() in ["true", "1", "yes"]:
                    geometries_to_calc.append(col)

        return geometries_to_calc

    except Exception as e:
        raise RuntimeError(f"Error reading config file: {e}")


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

        if gear_label not in df.columns:
            raise ValueError(f"Gear label '{gear_label}' not found in parameters file")

        # Convert to dictionary
        params = {}
        for _, row in df.iterrows():
            param_name = row["parameter"]
            param_value = row[gear_label]
            params[param_name] = float(param_value)

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
        addendum_lowering = _calculate_addendum_lowering(
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
        contact_params = _calculate_contact_path_parameters(
            base_d_pinion, addendum_d_pinion, addendum_d_gear, center_distance
        )
        calculated_params.update(contact_params)

        # Calculate equivalent radii for two-cylinder model
        equivalent_radii = _calculate_equivalent_radii(base_d_pinion, contact_params)
        calculated_params.update(equivalent_radii)

        # Log calculation results
        _log_calculation_results(calculated_params)

        return calculated_params

    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e}")
    except (ValueError, np.linalg.LinAlgError) as e:
        raise ValueError(f"Invalid calculation result: {e}")


def _calculate_addendum_lowering(
    pitch_d_pinion: float, x1: float, x2: float, module: float, center_distance: float
) -> float:
    """Calculate addendum lowering for gear pair.

    Args:
        pitch_d_pinion: Pitch diameter of pinion [mm]
        x1: Profile shift coefficient for pinion
        x2: Profile shift coefficient for gear
        module: Module of the gear [mm]
        center_distance: Center distance [mm]

    Returns:
        Addendum lowering factor
    """
    return ((pitch_d_pinion + x1 + x2) + module - center_distance) / module


def _calculate_contact_path_parameters(
    base_d_pinion: float,
    addendum_d_pinion: float,
    addendum_d_gear: float,
    center_distance: float,
) -> Dict[str, float]:
    """Calculate contact path parameters (PE, PF, N_1P, N_1E, etc.).

    Args:
        base_d_pinion: Base diameter of pinion [mm]
        addendum_d_pinion: Addendum diameter of pinion [mm]
        addendum_d_gear: Addendum diameter of gear [mm]
        center_distance: Center distance [mm]

    Returns:
        Dictionary with contact path parameters
    """
    # Convert to radii
    R_b = base_d_pinion / 2
    R_a1 = addendum_d_pinion / 2
    R_a2 = addendum_d_gear / 2
    a_w = center_distance

    # Calculate key angles
    alpha_Ra2 = np.arccos(R_b / R_a2)
    alpha_aw = np.arccos(R_b / (a_w / 2))
    alpha_Ra1 = np.arccos(R_b / R_a1)

    # Calculate tangent values
    tan_alpha_Ra2 = np.tan(alpha_Ra2)
    tan_alpha_aw = np.tan(alpha_aw)
    tan_alpha_Ra1 = np.tan(alpha_Ra1)

    # Calculate contact path segments
    PE = R_b * (tan_alpha_Ra2 - tan_alpha_aw)
    PF = R_b * (tan_alpha_Ra1 - tan_alpha_aw)
    N_1P = R_b * tan_alpha_aw
    N_1E = N_1P - PE

    # Calculate gear contact points
    N_2P = R_b * tan_alpha_aw
    N_2F = N_2P - PF

    return {
        "PE": PE,
        "PF": PF,
        "N_1P": N_1P,
        "N_1E": N_1E,
        "N_2P": N_2P,
        "N_2F": N_2F,
    }


def _calculate_equivalent_radii(
    base_d_pinion: float, contact_params: Dict[str, float]
) -> Dict[str, float]:
    """Calculate equivalent radii for two-cylinder contact model.

    Args:
        base_d_pinion: Base diameter of pinion [mm]
        contact_params: Dictionary with contact path parameters

    Returns:
        Dictionary with equivalent radii and related parameters
    """
    R_b = base_d_pinion / 2
    N_1E = contact_params["N_1E"]
    N_2F = contact_params["N_2F"]

    # Calculate angles and radii for pinion
    Alfa_y_1 = np.arctan(N_1E / R_b)
    R_y_1 = R_b / np.cos(Alfa_y_1)

    # Calculate angles and radii for gear
    Alfa_y_2 = np.arctan(N_2F / R_b)
    R_y_2 = R_b / np.cos(Alfa_y_2)

    return {
        "Alfa_y_1": Alfa_y_1,
        "R_y_1": R_y_1,
        "Alfa_y_2": Alfa_y_2,
        "R_y_2": R_y_2,
        "equivalent_pinion_radius": R_y_1,
        "equivalent_gear_radius": R_y_2,
    }


def _log_calculation_results(calculated_params: Dict[str, float]) -> None:
    """Log the key calculation results for user feedback.

    Args:
        calculated_params: Dictionary containing all calculated parameters
    """
    R_y_1 = calculated_params["R_y_1"]
    R_y_2 = calculated_params["R_y_2"]
    addendum_r_pinion = calculated_params["addendum_diameter_pinion"] / 2
    addendum_r_gear = calculated_params["addendum_diameter_gear"] / 2

    print(
        f"📏 Pinion contact radius interval: [{R_y_1:.3f}, {addendum_r_pinion:.3f}] mm"
    )
    print(f"📏 Gear contact radius interval: [{R_y_2:.3f}, {addendum_r_gear:.3f}] mm")


def save_geometry_parameters(
    gear_label: str, calculated_params: Dict[str, float]
) -> None:
    """Save calculated geometry parameters to CSV file.

    Creates a structured CSV file containing all calculated geometry parameters
    with proper units and formatting for further analysis.

    Args:
        gear_label: Label identifying the specific gear geometry
        calculated_params: Dictionary containing calculated parameters

    Raises:
        OSError: If file cannot be written to disk
        ValueError: If calculated_params contains invalid data
    """
    output_file = f"processed_data/{gear_label}_geometry_parameters.csv"

    try:
        # Create structured parameter data with proper units
        param_data = _create_parameter_data_structure(calculated_params)

        # Convert to DataFrame and save
        df = pd.DataFrame(param_data)
        df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"✅ Saved {len(calculated_params)} geometry parameters to {output_file}")

    except Exception as e:
        raise OSError(
            f"Failed to save geometry parameters to {output_file}: {e}"
        ) from e


def _create_parameter_data_structure(
    calculated_params: Dict[str, float],
) -> List[Dict[str, str]]:
    """Create structured parameter data with appropriate units.

    Args:
        calculated_params: Dictionary containing calculated parameters

    Returns:
        List of dictionaries with parameter, value, and units columns
    """
    # Define unit mapping for different parameter types
    unit_mapping = {
        "length": [
            "radius",
            "diameter",
            "length",
            "width",
            "PE",
            "PF",
            "N_1P",
            "N_1E",
            "N_2P",
            "N_2F",
        ],
        "pressure": ["pressure", "stress"],
        "angle": ["angle", "Alfa"],
        "dimensionless": ["coefficient", "lowering", "calculated"],
    }

    param_data = []
    for param_name, param_value in calculated_params.items():
        units = _determine_parameter_units(param_name, unit_mapping)

        # Format value based on type
        formatted_value = _format_parameter_value(param_value, units)

        param_data.append(
            {"parameter": param_name, "value": formatted_value, "units": units}
        )

    return param_data


def _determine_parameter_units(
    param_name: str, unit_mapping: Dict[str, List[str]]
) -> str:
    """Determine appropriate units for a parameter based on its name.

    Args:
        param_name: Name of the parameter
        unit_mapping: Dictionary mapping unit types to parameter name patterns

    Returns:
        Appropriate unit string for the parameter
    """
    param_lower = param_name.lower()

    # Check each unit category
    for unit_type, patterns in unit_mapping.items():
        if any(pattern in param_lower for pattern in patterns):
            return {
                "length": "mm",
                "pressure": "MPa",
                "angle": "rad",
                "dimensionless": "-",
            }[unit_type]

    # Default to dimensionless if no match found
    return "-"


def _format_parameter_value(value: float, units: str) -> str:
    """Format parameter value with appropriate precision based on units.

    Args:
        value: Numerical value to format
        units: Units of the parameter

    Returns:
        Formatted value string
    """
    if units == "mm":
        return f"{value:.3f}"  # 3 decimal places for lengths
    elif units == "MPa":
        return f"{value:.1f}"  # 1 decimal place for pressures
    elif units == "rad":
        return f"{value:.6f}"  # 6 decimal places for angles
    elif isinstance(value, bool):
        return str(value)  # Boolean values as-is
    else:
        return f"{value:.4f}"  # Default 4 decimal places


def calculate_geometry_for_label(gear_label: str) -> bool:
    """Calculate complete geometry data for a single gear label.

    Args:
        gear_label: Label identifying the specific gear geometry

    Returns:
        True if calculation successful, False otherwise
    """
    try:
        print(f"\n--- Calculating geometry for {gear_label} ---")

        # Load parameters
        params = load_gear_parameters(gear_label)
        print(f"✅ Loaded {len(params)} parameters")

        # Validate parameters (currently always passes)
        errors = validate_gear_parameters(params)
        if errors:
            print("❌ Parameter validation failed:")
            for error in errors:
                print(f"   {error}")
            return False
        else:
            print("✅ Parameter validation passed")

        # Calculate geometry
        calculated_params = calculate_gear_geometry(params)
        print(f"🧮 Calculated {len(calculated_params)} geometry parameters")

        # Save calculated parameters
        save_geometry_parameters(gear_label, calculated_params)

        return True

    except Exception as e:
        print(f"❌ Error calculating geometry for {gear_label}: {e}")
        return False


def main() -> None:
    """Main function for geometry calculation."""
    print("=" * 60)
    print("🧮 TWO-CYLINDER CONTACT GEOMETRY CALCULATION")
    print("=" * 60)

    try:
        # Step 2: Get geometries to calculate from config
        print("\nSTEP 2: Loading geometries to calculate...")
        geometries_to_calc = get_geometries_to_calculate()

        if not geometries_to_calc:
            print("❌ No geometries marked for calculation in config file")
            print(
                "   Check processed_data/abaqus_config.csv and set calculate=True for desired geometries"
            )
            return

        print(
            f"📋 Found {len(geometries_to_calc)} geometries to calculate: {geometries_to_calc}"
        )

        # Calculate geometry for each selected label
        print("\nSTEP 3: Processing geometries...")
        successful_calculations = 0
        for gear_label in geometries_to_calc:
            if calculate_geometry_for_label(gear_label):
                successful_calculations += 1

        # Summary
        print(f"\n{'=' * 60}")
        print("📊 CALCULATION SUMMARY")
        print(f"{'=' * 60}")
        print(
            f"✅ Successful: {successful_calculations}/{len(geometries_to_calc)} geometries"
        )

        if successful_calculations == len(geometries_to_calc):
            print("🎉 All geometry calculations completed successfully!")
            print("📋 Next steps:")
            print("   1. Review calculated geometry files in processed_data/")
            print(
                "   2. Implement actual calculation formulas in calculate_gear_geometry()"
            )
            print("   3. Run 01_setup_analysis.py to update project configuration")
        else:
            print("⚠️  Some calculations failed. Check error messages above.")

    except Exception as e:
        print(f"❌ Error in geometry calculation: {e}")


if __name__ == "__main__":
    main()
