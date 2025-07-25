import csv
import os


def matlab_to_csv(
    gears: list[str],
    csv_file_path: str = "initial_conditions/measured_deformation.csv",
    contact_width_prefix: str = "initial_conditions/AA_EXP",
    g_alpha_prefix: str = "initial_conditions/g_alpha",
) -> None:
    """Processes gear data from text files and writes it to a CSV file, preserving existing data.

    Args:
        gears (list[str]): A list of gear labels to process.
        csv_file_path (str, optional): The path to the CSV file to write to. Defaults to "initial_conditions/measured_deformation.csv".
        contact_width_prefix (str, optional): The prefix for contact width text files. Defaults to "initial_conditions/AA_EXP".
        g_alpha_prefix (str, optional): The prefix for g_alpha text files. Defaults to "initial_conditions/g_alpha".
    """
    # Read existing data if file exists
    existing_data = {}

    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            try:
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        geometry_id = row[0]
                        g_alpha = float(row[1])
                        contact_width = float(row[2])
                        if geometry_id not in existing_data:
                            existing_data[geometry_id] = []
                        existing_data[geometry_id].append((g_alpha, contact_width))
            except StopIteration:
                # File is empty, no existing data to read
                pass

        if existing_data:
            print(
                f"Found existing data for {len(existing_data)} geometries in CSV file - preserving existing data"
            )

    # Write to measured_deformation.csv in row format
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["geometry_id", "g_alpha", "contact_width_S"])

        # First, write existing data back
        for geometry_id, data_pairs in existing_data.items():
            for g_val, contact_val in data_pairs:
                writer.writerow([geometry_id, g_val, contact_val])

        for gear in gears:
            gear_num = gear.split("_")[1]

            try:
                # Read contact_width values
                with open(f"{contact_width_prefix}_{gear_num}.txt", "r") as f:
                    line = f.readline()
                    values = [float(x) for x in line.strip().split("\t")]

                # Read g_alpha values
                with open(f"{g_alpha_prefix}_{gear_num}.txt", "r") as f:
                    line = f.readline()
                    g_alpha = [float(x) for x in line.strip().split("\t")]

                if len(g_alpha) != len(values):
                    print(f"Warning: Length mismatch for {gear}, skipping")
                    continue

                # Create new data pairs for this gear
                new_data_pairs = list(zip(g_alpha, values, strict=False))

                # Check if this gear already exists with the same data
                if gear in existing_data and existing_data[gear] == new_data_pairs:
                    print(f"Skipping {gear}: already processed with same data")
                    continue

                # Write each pair as a separate row
                for g_val, contact_val in new_data_pairs:
                    writer.writerow([gear, g_val, contact_val])

            except FileNotFoundError as e:
                print(f"File not found for {gear}: {e}")
                continue
            except ValueError as e:
                print(f"Error parsing data for {gear}: {e}")
                continue
