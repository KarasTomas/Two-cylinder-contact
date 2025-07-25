from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_addendum_lowering(
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


def calculate_contact_path_parameters(
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


def calculate_equivalent_radii(
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


def log_calculation_results(calculated_params: Dict[str, float]) -> None:
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
        param_data = create_parameter_data_structure(calculated_params)

        # Convert to DataFrame and save
        df = pd.DataFrame(param_data)
        df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"✅ Saved {len(calculated_params)} geometry parameters to {output_file}")

    except Exception as e:
        raise OSError(
            f"Failed to save geometry parameters to {output_file}: {e}"
        ) from e


def create_parameter_data_structure(
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
        units = determine_parameter_units(param_name, unit_mapping)

        # Format value based on type
        formatted_value = format_parameter_value(param_value, units)

        param_data.append(
            {"parameter": param_name, "value": formatted_value, "units": units}
        )

    return param_data


def determine_parameter_units(
    param_name: str, unit_mapping: Dict[str, List[str]]
) -> str:
    """Determine appropriate units for a parameter based on its name."""
    param_lower = param_name.lower()

    # Explicit mapping for known parameters
    explicit_units = {
        "pitch_diameter_pinion": "mm",
        "pitch_diameter_gear": "mm",
        "base_diameter_pinion": "mm",
        "base_diameter_gear": "mm",
        "addendum_diameter_pinion": "mm",
        "addendum_diameter_gear": "mm",
        "addendum_lowering": "-",
        "rolling_pressure_angle": "rad",
        "PE": "mm",
        "PF": "mm",
        "N_1P": "mm",
        "N_1E": "mm",
        "N_2P": "mm",
        "N_2F": "mm",
        "Alfa_y_1": "rad",
        "R_y_1": "mm",
        "Alfa_y_2": "rad",
        "R_y_2": "mm",
        "equivalent_pinion_radius": "mm",
        "equivalent_gear_radius": "mm",
    }
    # Try explicit mapping first
    if param_name in explicit_units:
        return explicit_units[param_name]

    # Fallback to substring mapping
    for unit_type, patterns in unit_mapping.items():
        if any(pattern in param_lower for pattern in patterns):
            return {
                "length": "mm",
                "pressure": "MPa",
                "angle": "rad",
                "dimensionless": "-",
            }[unit_type]

    return "-"


def format_parameter_value(value: float, units: str) -> str:
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
