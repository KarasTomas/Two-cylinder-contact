# Tools for extracting data from the two-cylinder contact simulation result files.
import csv
import os
import time

from abaqus import *
from abaqusConstants import *

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
GEOMETRY_SUFFIX = "179"
SELECTED_VARIABLES = ["U2", "CPRESS", "CFNORM"]  # Variables to extract

# Path and extraction settings
NUM_INTERVALS = 50  # Number of points along extraction paths
SKIP_EXISTING_CSV = True  # Skip extraction if CSV already exists
FORCE_REEXTRACT = False  # Re-extract even if CSV exists

# Variable configuration (can include more than selected)
# Variable configuration (updated with correct syntax)
VARIABLE_CONFIG = {
    "U2": {
        "variableLabel": "U",
        "outputPosition": NODAL,
        "refinement": (COMPONENT, "U2"),
        "description": "Y-displacement",
    },
    "CPRESS": {
        "variableLabel": "CPRESS",
        "outputPosition": ELEMENT_NODAL,
        "description": "Contact pressure",
    },
    "CFNORM": {
        "variableLabel": "CNORMF",
        "outputPosition": ELEMENT_NODAL,
        "refinement": (COMPONENT, "CNORMF2"),
        "description": "Contact normal force",
    },
    "S": {
        "variableLabel": "S",
        "outputPosition": INTEGRATION_POINT,
        "description": "Stress components",
    },
    "RF": {
        "variableLabel": "RF",
        "outputPosition": NODAL,
        "description": "Reaction forces",
    },
    "U": {
        "variableLabel": "U",
        "outputPosition": NODAL,
        "description": "Displacement components",
    },
    "CSTRESS": {
        "variableLabel": "CSTRESS",
        "outputPosition": ELEMENT_NODAL,
        "description": "Contact stress",
    },
}

# Path configuration
PATH_CONFIG = {
    "pinion": {
        "instance": "PINION-1",
        "node_set": "PINION_EDGE",
        "edge_spec": (1, 4, -1),  # (start_edge, end_edge, direction)
    },
    "gear": {
        "instance": "GEAR-1",
        "node_set": "GEAR_EDGE",
        "edge_spec": (1, 2, 1),  # (start_edge, end_edge, direction)
    },
}
# ==============================================================================


def get_database_path():
    """Get the path to the model database file"""
    database_path = os.path.join(
        "..",
        "results",
        "Gears_" + GEOMETRY_SUFFIX + "_results",
        "Gears_" + GEOMETRY_SUFFIX + "_model_database.csv",
    )
    return database_path


def get_results_paths():
    """Get paths for ODB files and extracted data"""
    base_path = os.path.join("..", "results", "Gears_" + GEOMETRY_SUFFIX + "_results")
    paths = {
        "odb_files": os.path.join(base_path, "odb_files"),
        "extracted_data": os.path.join(base_path, "extracted_data"),
        "database": os.path.join(
            base_path, "Gears_" + GEOMETRY_SUFFIX + "_model_database.csv"
        ),
    }
    return paths


def read_database():
    """Read the model database CSV file"""
    database_path = get_database_path()

    if not os.path.exists(database_path):
        print("ERROR: Model database not found at: " + database_path)
        return []

    try:
        with open(database_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            data = []
            for row in reader:
                data.append(row)
            return data
    except Exception as e:
        print("ERROR reading model database: " + str(e))
        return []


def find_jobs_to_extract():
    """Find jobs that need data extraction"""
    print("=" * 60)
    print("SCANNING FOR JOBS TO EXTRACT")
    print("=" * 60)

    database = read_database()
    if not database:
        return []

    jobs_to_extract = []

    for row in database:
        # Check if job has been moved and is ready for extraction
        if row.get("odb_moved", "").lower() == "true":
            # Get job_name from odb_name (remove .odb extension)
            odb_name = row.get("odb_name", "")
            job_name = (
                odb_name.replace(".odb", "") if odb_name.endswith(".odb") else odb_name
            )

            # Skip if already extracted (unless force re-extract)
            if not FORCE_REEXTRACT and row.get("result_csv_file", ""):
                print("  SKIPPED (already extracted): " + job_name)
                continue

            # Add job_name to the row for easier access later
            row["job_name"] = job_name
            jobs_to_extract.append(row)
            print("  FOUND: " + job_name)
        else:
            # Get job_name for display even if not processing
            odb_name = row.get("odb_name", "")
            job_name = (
                odb_name.replace(".odb", "") if odb_name.endswith(".odb") else odb_name
            )
            print("  SKIPPED (not moved): " + job_name)

    print("\nSummary:")
    print("  Total jobs in database: " + str(len(database)))
    print("  Jobs ready for extraction: " + str(len(jobs_to_extract)))

    return jobs_to_extract


def create_edge_path(odb, component_name, instance_name, node_set_name, edge_spec):
    """Create extraction path using edge list method"""
    try:
        # Get the instance and node set
        instance = odb.rootAssembly.instances[instance_name]
        nodes = odb.rootAssembly.instances[instance_name].nodeSets[node_set_name].nodes

        # Build set of node labels for fast lookup
        nodeLabels = set([node.label for node in nodes])

        # Find elements that have at least 2 nodes in the node set
        elemsInNodeSet = []
        for elem in instance.elements:
            count = sum(
                [1 for nodeLabel in elem.connectivity if nodeLabel in nodeLabels]
            )
            if count >= 2:
                elemsInNodeSet.append(elem)

        # Sort elements by label for consistency
        elemsInNodeSet.sort(key=lambda elem: elem.label)

        # Create edge list
        edgeList = []
        start_edge, end_edge, direction = edge_spec
        for elem in elemsInNodeSet:
            edgeList.append((elem.label, start_edge, end_edge, direction))

        if not edgeList:
            print("    ERROR: No valid edges found for " + component_name)
            return None

        # Create path name
        path_name = component_name + "_path_" + str(int(time.time()))

        # Create the path
        path = session.Path(
            name=path_name, type=EDGE_LIST, expression=[(instance_name, edgeList)]
        )

        print("    SUCCESS: Created path with " + str(len(edgeList)) + " edges")
        return path

    except Exception as e:
        print("    ERROR creating path: " + str(e))
        return None


def extract_variable_data(odb, variable_name, path, component_name):
    """Extract variable data along specified path"""
    try:
        # Get variable configuration
        if variable_name not in VARIABLE_CONFIG:
            print("      ERROR: Unknown variable " + variable_name)
            return None

        var_config = VARIABLE_CONFIG[variable_name]

        # Set primary variable with correct syntax
        if "refinement" in var_config:
            # For variables that need refinement (like U2, CNORMF2)
            session.viewports["Viewport: 1"].odbDisplay.setPrimaryVariable(
                variableLabel=var_config["variableLabel"],
                outputPosition=var_config["outputPosition"],
                refinement=var_config["refinement"],
            )
        else:
            # For variables without refinement (like CPRESS)
            session.viewports["Viewport: 1"].odbDisplay.setPrimaryVariable(
                variableLabel=var_config["variableLabel"],
                outputPosition=var_config["outputPosition"],
            )

        # Create unique XY data name
        xy_data_name = (
            variable_name + "_" + component_name + "_" + str(int(time.time()))
        )

        # Extract XY data using UNDEFORMED shape (as in your macro)
        session.XYDataFromPath(
            name=xy_data_name,
            path=path,
            includeIntersections=False,
            projectOntoMesh=False,
            pathStyle=PATH_POINTS,
            numIntervals=NUM_INTERVALS,
            projectionTolerance=0,
            shape=UNDEFORMED,  # Changed from DEFORMED to UNDEFORMED
            labelType=TRUE_DISTANCE,
            removeDuplicateXYPairs=True,
            includeAllElements=False,
        )

        # Get the data
        xy_data = session.xyDataObjects[xy_data_name].data

        # Clean up XY data object
        del session.xyDataObjects[xy_data_name]

        print("      SUCCESS: Extracted " + str(len(xy_data)) + " data points")
        return xy_data

    except Exception as e:
        print("      ERROR extracting " + variable_name + ": " + str(e))
        return None


def extract_all_data_from_odb(job_info):
    """Extract all selected variables from a single ODB file"""
    job_name = job_info["job_name"]

    print("\n" + "=" * 60)
    print("EXTRACTING DATA FROM: " + job_name)
    print("=" * 60)

    # Construct ODB path
    paths = get_results_paths()
    odb_path = os.path.join(paths["odb_files"], job_name + ".odb")

    if not os.path.exists(odb_path):
        print("ERROR: ODB file not found: " + odb_path)
        return None, []

    extraction_results = {
        "job_name": job_name,
        "successful_variables": [],
        "data": {},
        "errors": [],
    }

    try:
        # Open ODB
        print("Opening ODB file...")
        odb = session.openOdb(name=odb_path)
        session.viewports["Viewport: 1"].setValues(displayedObject=odb)

        # Create paths for both components
        paths_created = {}

        for component_name, config in PATH_CONFIG.items():
            print("  Creating " + component_name + " path...")
            path = create_edge_path(
                odb,
                component_name,
                config["instance"],
                config["node_set"],
                config["edge_spec"],
            )
            if path:
                paths_created[component_name] = path
            else:
                extraction_results["errors"].append(
                    "Path creation failed for " + component_name
                )

        if not paths_created:
            print("ERROR: No paths created - cannot extract data")
            odb.close()
            return None, []

        # Extract variables
        for variable_name in SELECTED_VARIABLES:
            print("  Extracting variable: " + variable_name)

            variable_data = {}
            variable_success = False

            for component_name, path in paths_created.items():
                print("    From " + component_name + "...")
                data = extract_variable_data(odb, variable_name, path, component_name)

                if data:
                    # Convert to lists for CSV writing
                    x_data = [point[0] for point in data]
                    y_data = [point[1] for point in data]
                    variable_data[component_name] = {"x": x_data, "y": y_data}
                    variable_success = True
                else:
                    extraction_results["errors"].append(
                        "Failed to extract " + variable_name + " from " + component_name
                    )

            if variable_success:
                extraction_results["data"][variable_name] = variable_data
                extraction_results["successful_variables"].append(variable_name)
                print("    VARIABLE SUCCESS: " + variable_name)
            else:
                print("    VARIABLE FAILED: " + variable_name)

        # Clean up
        for path_name in [path.name for path in paths_created.values()]:
            try:
                del session.paths[path_name]
            except:
                pass

        odb.close()

        print("\nExtraction Summary:")
        print(
            "  Successful variables: "
            + str(len(extraction_results["successful_variables"]))
        )
        print("  Errors: " + str(len(extraction_results["errors"])))

        return extraction_results, extraction_results["successful_variables"]

    except Exception as e:
        print("ERROR during extraction: " + str(e))
        try:
            odb.close()
        except:
            pass
        return None, []


def save_to_csv(extraction_results, csv_path):
    """Save extracted data to CSV file"""
    if not extraction_results or not extraction_results["data"]:
        print("ERROR: No data to save")
        return False

    try:
        print("Saving data to CSV: " + csv_path)

        # Prepare CSV data structure
        csv_data = {}
        max_rows = 0

        # Organize data by columns
        for variable_name, variable_data in extraction_results["data"].items():
            for component_name, component_data in variable_data.items():
                x_col = "X_" + component_name + "_" + variable_name
                y_col = "Y_" + component_name + "_" + variable_name

                csv_data[x_col] = component_data["x"]
                csv_data[y_col] = component_data["y"]

                max_rows = max(max_rows, len(component_data["x"]))

        # Ensure all columns have same length (pad with empty strings)
        for col_name, col_data in csv_data.items():
            while len(col_data) < max_rows:
                col_data.append("")

        # Write CSV file
        with open(csv_path, "wb") as csvfile:
            fieldnames = sorted(csv_data.keys())  # Sort column names for consistency
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write data rows
            for row_idx in range(max_rows):
                row_data = {}
                for col_name in fieldnames:
                    row_data[col_name] = (
                        csv_data[col_name][row_idx]
                        if row_idx < len(csv_data[col_name])
                        else ""
                    )
                writer.writerow(row_data)

        print(
            "SUCCESS: CSV saved with "
            + str(max_rows)
            + " rows and "
            + str(len(fieldnames))
            + " columns"
        )
        return True

    except Exception as e:
        print("ERROR saving CSV: " + str(e))
        return False


def update_database_with_results(job_name, csv_filename, successful_variables):
    """Update database with extraction results"""
    database_path = get_database_path()

    try:
        # Read current database
        data = []
        fieldnames = []

        with open(database_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames

            # Add new columns if they don't exist
            if "result_csv_file" not in fieldnames:
                fieldnames.append("result_csv_file")
            if "result_content" not in fieldnames:
                fieldnames.append("result_content")

            for row in reader:
                # Get job_name from odb_name for comparison
                odb_name = row.get("odb_name", "")
                odb_job_name = (
                    odb_name.replace(".odb", "")
                    if odb_name.endswith(".odb")
                    else odb_name
                )

                # Update the matching job
                if odb_job_name == job_name:
                    row["result_csv_file"] = csv_filename
                    row["result_content"] = str(successful_variables)

                # Add missing fields
                if "result_csv_file" not in row:
                    row["result_csv_file"] = ""
                if "result_content" not in row:
                    row["result_content"] = ""

                data.append(row)

        # Write updated database
        with open(database_path, "wb") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print("Database updated: " + job_name + " -> " + csv_filename)
        return True

    except Exception as e:
        print("ERROR updating database: " + str(e))
        return False


def generate_extraction_log(processed_jobs, successful_extractions, failed_extractions):
    """Generate extraction log file"""
    paths = get_results_paths()
    log_path = os.path.join(paths["extracted_data"], "extraction_log.txt")

    try:
        with open(log_path, "w") as logfile:
            logfile.write("Abaqus Results Extraction Log\n")
            logfile.write("=" * 50 + "\n")
            logfile.write("Geometry: " + GEOMETRY_SUFFIX + "\n")
            logfile.write(
                "Extraction time: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
            logfile.write("Selected variables: " + str(SELECTED_VARIABLES) + "\n")
            logfile.write("\n")

            logfile.write("Summary:\n")
            logfile.write("  Total jobs processed: " + str(len(processed_jobs)) + "\n")
            logfile.write(
                "  Successful extractions: " + str(len(successful_extractions)) + "\n"
            )
            logfile.write(
                "  Failed extractions: " + str(len(failed_extractions)) + "\n"
            )
            logfile.write("\n")

            if successful_extractions:
                logfile.write("Successful Extractions:\n")
                for job_name, variables in successful_extractions.items():
                    logfile.write("  " + job_name + ": " + str(variables) + "\n")
                logfile.write("\n")

            if failed_extractions:
                logfile.write("Failed Extractions:\n")
                for job_name, error in failed_extractions.items():
                    logfile.write("  " + job_name + ": " + error + "\n")

        print("Extraction log saved: " + log_path)

    except Exception as e:
        print("ERROR creating extraction log: " + str(e))


def main():
    """Main extraction function"""
    print("=" * 60)
    print("ABAQUS RESULTS EXTRACTION")
    print("=" * 60)
    print("Geometry: " + GEOMETRY_SUFFIX)
    print("Variables: " + str(SELECTED_VARIABLES))
    print("Force re-extract: " + str(FORCE_REEXTRACT))

    # Find jobs to extract
    jobs_to_extract = find_jobs_to_extract()

    if not jobs_to_extract:
        print("\nNo jobs found for extraction")
        print("Make sure to run move_results.py first")
        return

    # Create extracted_data directory if it doesn't exist
    paths = get_results_paths()
    if not os.path.exists(paths["extracted_data"]):
        os.makedirs(paths["extracted_data"])
        print("Created directory: " + paths["extracted_data"])

    # Process each job
    processed_jobs = []
    successful_extractions = {}
    failed_extractions = {}

    for i, job_info in enumerate(jobs_to_extract):
        job_name = job_info["job_name"]

        print("\n" + "=" * 60)
        print("PROCESSING JOB " + str(i + 1) + "/" + str(len(jobs_to_extract)))
        print("=" * 60)

        # Extract data
        extraction_results, successful_variables = extract_all_data_from_odb(job_info)
        processed_jobs.append(job_name)

        if extraction_results and successful_variables:
            # Save to CSV
            csv_filename = job_name + ".csv"
            csv_path = os.path.join(paths["extracted_data"], csv_filename)

            if save_to_csv(extraction_results, csv_path):
                # Update database
                update_database_with_results(
                    job_name, csv_filename, successful_variables
                )
                successful_extractions[job_name] = successful_variables
                print("SUCCESS: " + job_name)
            else:
                failed_extractions[job_name] = "CSV save failed"
                print("FAILED: " + job_name + " (CSV save failed)")
        else:
            failed_extractions[job_name] = "Data extraction failed"
            print("FAILED: " + job_name + " (data extraction failed)")

    # Generate final summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("Total jobs processed: " + str(len(processed_jobs)))
    print("Successful extractions: " + str(len(successful_extractions)))
    print("Failed extractions: " + str(len(failed_extractions)))

    if successful_extractions:
        print("\nSuccessful extractions:")
        for job_name, variables in successful_extractions.items():
            print("  " + job_name + ": " + str(variables))

    if failed_extractions:
        print("\nFailed extractions:")
        for job_name, error in failed_extractions.items():
            print("  " + job_name + ": " + error)

    # Generate extraction log
    generate_extraction_log(processed_jobs, successful_extractions, failed_extractions)

    print("\nResults saved to: " + paths["extracted_data"])
    print(
        "Extraction log: " + os.path.join(paths["extracted_data"], "extraction_log.txt")
    )


if __name__ == "__main__":
    main()
