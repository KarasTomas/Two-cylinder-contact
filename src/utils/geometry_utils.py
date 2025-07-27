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
    return (pitch_d_pinion + (x1 + x2) * module - center_distance) / module


def calculate_contact_path_parameters(
    base_d_pinion: float,
    addendum_d_pinion: float,
    addendum_d_gear: float,
    center_distance: float,
    module: float,
    pressure_angle: float,
) -> Dict[str, float]:
    """Calculate contact path parameters (PE, PF, N_1P, N_1E, etc.).

    Args:
        base_d_pinion: Base diameter of pinion [mm]
        addendum_d_pinion: Addendum diameter of pinion [mm]
        addendum_d_gear: Addendum diameter of gear [mm]
        center_distance: Center distance [mm]
        module: Module of the gear [mm]
        pressure_angle: Pressure angle [rad]

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

    # Calculate path of contact
    g_alpha = PE + PF

    # Calculate profile contact ratio
    epsilon_alpha = g_alpha / (np.pi * module * np.cos(pressure_angle))

    return {
        "PE": PE,
        "PF": PF,
        "N_1P": N_1P,
        "N_1E": N_1E,
        "N_2P": N_2P,
        "N_2F": N_2F,
        "g_alpha": g_alpha,
        "epsilon_alpha": epsilon_alpha,
    }


def calculate_min_equivalent_radii(
    base_d_pinion: float, contact_params: Dict[str, float]
) -> Dict[str, float]:
    """Calculate minimal equivalent radii for two-cylinder contact model.

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


def calculate_equivalent_rho(
    R_b: tuple[float, float],
    NE: tuple[float, float],
    contact_path: List[float],
    rho_names: tuple[str, str] = ("pinion", "gear"),
) -> dict[str, List[float]]:
    """Generate an array of radii and corresponding rho values for the two-cylinder contact model.

    Args:
        R_b: Base radius [mm]
        NE: Distance from point N to start of contact path (point E or F) [mm]
        contact_path: List of contact path points [mm]
        rho_names: Names for rho values (default: ("pinion", "gear"))

    Returns:
        rho: List of rho values corresponding to the radii
    """
    eq_rho = {}
    eq_rho["contact_path"] = contact_path
    for r_b, n1e, rho_name in zip(R_b, NE, rho_names, strict=False):
        alpha_y = np.arctan((n1e + np.array(contact_path)) / r_b)
        R_y = r_b / np.cos(alpha_y)
        rho = r_b * np.tan(alpha_y)
        eq_rho[f"rho_{rho_name}"] = rho.tolist()
        eq_rho[f"R_y_{rho_name}"] = R_y.tolist()

    # Flip the second rho array (usually for the gear)
    if len(rho_names) > 1:
        eq_rho[f"rho_{rho_names[1]}"] = eq_rho[f"rho_{rho_names[1]}"][::-1]
    return eq_rho


def load_measured_deformation(
    gear_label: str, file_path: str = "initial_conditions/measured_deformation.csv"
) -> dict[str, list[float]]:
    """Load measured deformation data for a given gear_label from CSV.

    Returns:
        Tuple of (g_alpha: list[float], contact_width_S: list[float])
    """
    df = pd.read_csv(file_path)
    df_gear = df[df["geometry_id"] == gear_label]
    measured_deformation = (
        df_gear[["g_alpha", "contact_width_S"]].astype(float).to_dict(orient="list")
    )
    return measured_deformation


def map_deformation_to_contact_path(
    eq_rho: dict, measured_deformation: dict
) -> np.ndarray:
    """Interpolate measured deformation (contact_width_S) onto the eq_rho contact_path grid.

    Args:
        eq_rho: dict with "contact_path" (list of floats)
        measured_deformation: dict with "g_alpha" and "contact_width_S" (lists of floats)

    Returns:
        interp_contact_width_S: np.ndarray, interpolated to eq_rho["contact_path"]
    """
    contact_path = np.array(eq_rho["contact_path"])
    g_alpha_measured = np.array(measured_deformation["g_alpha"])
    contact_width_S_measured = np.array(measured_deformation["contact_width_S"])

    # Interpolate measured deformation onto the contact_path grid
    interp_contact_width_S = np.interp(
        contact_path, g_alpha_measured, contact_width_S_measured
    )

    return interp_contact_width_S


def calculate_deformation_loading(
    eq_rho: Dict[str, List[float]],
    gear_label: str,
    file_path: str = "initial_conditions/measured_deformation.csv",
) -> dict[str, List[float]]:
    """Calculate deformation loading for the two-cylinder contact model.

    Args:
        eq_rho: Dictionary containing equivalent radii and rho values
        gear_label: Label identifying the specific gear geometry
        file_path: Path to the measured deformation data file

    Returns:
        Dictionary containing deformation loading values
    """
    # Load measured deformation data
    measured_deformation = load_measured_deformation(gear_label, file_path)

    interp_contact_width_S = map_deformation_to_contact_path(
        eq_rho, measured_deformation
    )
    deformation_loading = {
        "contact_path": eq_rho["contact_path"],
        "deformation_loading": interp_contact_width_S,
    }
    return deformation_loading


def calculate_load_delta(sim_data: Dict[str, List[float]]) -> list[float]:
    """Calculate the load delta from deformation loading."""
    rho_1 = np.array(sim_data["rho_pinion"])
    rho_2 = np.array(sim_data["rho_gear"])
    contact_width_S = np.array(sim_data["deformation_loading"])

    h1 = rho_1 - 0.5 * np.sqrt(4 * rho_1**2 - contact_width_S**2)
    h2 = rho_2 - 0.5 * np.sqrt(4 * rho_2**2 - contact_width_S**2)

    load_delta = h1 + h2

    return load_delta.tolist()


def save_simulation_data(sim_data: Dict[str, List[float]], gear_label: str) -> None:
    """Save simulation data to a CSV file."""
    output_file = f"processed_data/{gear_label}_simulation_data.csv"

    try:
        df = pd.DataFrame(sim_data)
        df.to_csv(output_file, index=False, encoding="utf-8")

    except Exception as e:
        raise OSError(f"Failed to save simulation data to {output_file}: {e}") from e


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
    gear_label: str, input_params: Dict[str, float], calculated_params: Dict[str, float]
) -> None:
    """Save all geometry parameters (input + calculated) to CSV file.

    Args:
        gear_label: Label identifying the specific gear geometry
        input_params: Dictionary of input parameters (center_distance, etc.)
        calculated_params: Dictionary of calculated parameters

    Raises:
        OSError: If file cannot be written to disk
        ValueError: If parameters contain invalid data
    """
    output_file = f"processed_data/{gear_label}_geometry_parameters.csv"

    try:
        # Merge input and calculated parameters, input first
        all_params = {**input_params, **calculated_params}
        param_data = create_parameter_data_structure(all_params)

        df = pd.DataFrame(param_data)
        df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"✅ Saved {len(all_params)} geometry parameters to {output_file}")

    except Exception as e:
        raise OSError(
            f"Failed to save geometry parameters to {output_file}: {e}"
        ) from e


def create_parameter_data_structure(
    params: Dict[str, float],
) -> List[Dict[str, str]]:
    """Create structured parameter data with appropriate units.

    Args:
        params: Dictionary containing all parameters (input + calculated)

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
            "g_alpha",
        ],
        "pressure": ["pressure", "stress"],
        "angle": ["angle", "Alfa"],
        "dimensionless": [
            "coefficient",
            "lowering",
            "calculated",
            "profile contact ratio",
        ],
    }

    param_data = []
    for param_name, param_value in params.items():
        units = determine_parameter_units(param_name, unit_mapping)
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
        "center_distance": "mm",
        "pressure_angle": "°",
        "normal_module": "mm",
        "face_width": "mm",
        "tooth_count_pinion": "-",
        "tooth_count_gear": "-",
        "profile_shift_coefficient_pinion": "-",
        "profile_shift_coefficient_gear": "-",
        "elastic_modulus": "MPa",
        "poisson_ratio": "-",
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
        "g_alpha": "mm",
        "profile_contact_ratio": "-",
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
