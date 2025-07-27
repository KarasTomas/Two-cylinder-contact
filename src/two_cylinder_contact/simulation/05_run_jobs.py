# Tools for running the two-cylinder contact simulation jobs in a batch mode.
import csv
import os
import time

from abaqus import *
from abaqusConstants import *

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
GEOMETRY_SUFFIX = "179"  # Change this to match your geometry
BATCH_SIZE = 2  # Number of jobs to run simultaneously
WAIT_BETWEEN_SUBMISSIONS = 0  # Seconds to wait between job submissions
SKIP_EXISTING_ODBS = True  # Skip jobs if ODB already exists

# EXECUTION CONTROL
AUTO_START = False  # Set to True to start immediately without confirmation
PROCESS_ALL_BATCHES = True # Set to True to process all batches automatically
STOP_AFTER_BATCH = 3  # Stop after N batches (0 = process all)
PAUSE_BETWEEN_BATCHES = 0  # Seconds to pause between batches (0 = no pause)

# OUTPUT CONTROL
VERBOSE_STATUS = True  # Set to True for detailed status breakdown
# ==============================================================================


def get_database_path():
    """Get the path to the model database file"""
    return os.path.join(
        "..",
        "results",
        "Gears_" + GEOMETRY_SUFFIX + "_results",
        "Gears_" + GEOMETRY_SUFFIX + "_model_database.csv",
    )


def read_model_database():
    """Read the model database CSV file"""
    database_path = get_database_path()

    if not os.path.exists(database_path):
        return []

    try:
        with open(database_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            return [row for row in reader]
    except Exception as e:
        print("Error reading model database: " + str(e))
        return []


def update_job_status_in_database(job_name, status):
    """Update the odb_status for a specific job in the database"""
    database_path = get_database_path()

    if not os.path.exists(database_path):
        return

    try:
        # Read all data
        data = []
        fieldnames = []
        found_match = False

        with open(database_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames

            # Add odb_status column if it doesn't exist
            if "odb_status" not in fieldnames:
                fieldnames.append("odb_status")

            for row in reader:
                # Get odb_name and remove .odb extension for comparison
                odb_name = row.get("odb_name", "")
                odb_name_no_ext = (
                    odb_name.replace(".odb", "")
                    if odb_name.endswith(".odb")
                    else odb_name
                )

                # Update the status for matching job
                if odb_name_no_ext == job_name:
                    old_status = row.get("odb_status", "")
                    if old_status != status:
                        row["odb_status"] = status
                        found_match = True

                # Add odb_status field if missing
                if "odb_status" not in row:
                    row["odb_status"] = ""
                data.append(row)

        # Write back the updated data only if changes were made
        if found_match:
            with open(database_path, "wb") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

    except Exception as e:
        print("Error updating database: " + str(e))


def get_job_status_from_sta_file(job_name):
    """Get job status by reading the .sta file - most reliable method"""
    sta_paths = [
        job_name + ".sta",  # Current directory (abaqus_work)
        os.path.join(
            "..",
            "results",
            "Gears_" + GEOMETRY_SUFFIX + "_results",
            "odb_files",
            job_name + ".sta",
        ),
    ]

    for sta_path in sta_paths:
        if os.path.exists(sta_path):
            try:
                with open(sta_path, "r") as f:
                    content = f.read()

                # Check for completion indicators
                if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in content:
                    return "COMPLETED"
                elif "THE ANALYSIS HAS NOT BEEN COMPLETED" in content:
                    return "ABORTED"
                elif "ANALYSIS TERMINATED" in content:
                    return "TERMINATED"
                elif "ERROR" in content.upper():
                    return "ERROR"
                else:
                    return "INCOMPLETE"
            except:
                pass

    return None


def get_job_status_string(job_name):
    """Get current status of a job as string"""
    try:
        status = mdb.jobs[job_name].status
        status_str = str(status)

        if "." in status_str:
            return status_str.split(".")[-1]
        else:
            try:
                return status.name
            except:
                return status_str.upper()
    except:
        return "UNKNOWN"


def get_comprehensive_job_status(job_name):
    """Get comprehensive job status using multiple methods"""
    # Method 1: Check STA file (most reliable)
    sta_status = get_job_status_from_sta_file(job_name)

    # Method 2: Check job object status
    job_status = None
    try:
        if job_name in mdb.jobs:
            job_status = get_job_status_string(job_name)
    except:
        pass

    # Method 3: Check ODB existence
    odb_paths = [
        job_name + ".odb",
        os.path.join(
            "..",
            "results",
            "Gears_" + GEOMETRY_SUFFIX + "_results",
            "odb_files",
            job_name + ".odb",
        ),
    ]
    odb_exists = any(os.path.exists(path) for path in odb_paths)

    # Determine final status using priority logic
    if sta_status:
        if sta_status == "COMPLETED" and odb_exists:
            return "COMPLETED"
        elif sta_status in ["ABORTED", "TERMINATED", "ERROR"]:
            return sta_status
        else:
            return sta_status
    elif job_status:
        if job_status in ["RUNNING", "SUBMITTED"]:
            return "RUNNING"
        elif job_status == "COMPLETED" and odb_exists:
            return "COMPLETED"
        else:
            return job_status
    elif odb_exists:
        return "COMPLETED_NO_STA"
    else:
        return "PENDING"


def find_all_jobs():
    """Find all jobs by scanning for .inp files and checking CAE session"""
    # Find jobs from files
    search_dirs = [
        ".",
        os.path.join(
            "..", "results", "Gears_" + GEOMETRY_SUFFIX + "_results", "odb_files"
        ),
    ]

    found_jobs = set()
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for filename in os.listdir(search_dir):
                if filename.endswith(".inp") and filename.startswith(
                    "Gears_" + GEOMETRY_SUFFIX + "_"
                ):
                    job_name = filename[:-4]  # Remove .inp extension
                    found_jobs.add(job_name)

    # Filter to jobs available in current CAE session
    available_jobs = []
    for job_name in found_jobs:
        if job_name in mdb.jobs:
            job_info = {
                "name": job_name,
                "model": mdb.jobs[job_name].model,
                "odb_path": job_name + ".odb",
            }
            available_jobs.append(job_info)

    return list(found_jobs), available_jobs


def get_job_status_summary(jobs):
    """Get summary of job statuses"""
    status_counts = {
        "COMPLETED": 0,
        "ABORTED": 0,
        "RUNNING": 0,
        "PENDING": 0,
        "OTHER": 0,
    }

    for job in jobs:
        status = get_comprehensive_job_status(job["name"])

        if status == "COMPLETED":
            status_counts["COMPLETED"] += 1
        elif status in ["ABORTED", "TERMINATED", "ERROR"]:
            status_counts["ABORTED"] += 1
        elif status in ["RUNNING", "SUBMITTED"]:
            status_counts["RUNNING"] += 1
        elif status == "PENDING":
            status_counts["PENDING"] += 1
        else:
            status_counts["OTHER"] += 1

    return status_counts


def update_all_job_statuses(jobs):
    """Update status for all jobs in database"""
    updated_count = 0
    for job in jobs:
        job_name = job["name"]
        status = get_comprehensive_job_status(job_name)
        update_job_status_in_database(job_name, status)
        updated_count += 1

    if updated_count > 0:
        print("Database updated: " + str(updated_count) + " jobs")


def check_geometry_consistency():
    """Check if GEOMETRY_SUFFIX matches the jobs in the CAE file"""
    job_names = mdb.jobs.keys()

    if not job_names:
        return True, "No jobs found"

    # Extract geometry suffixes from job names
    found_suffixes = set()
    for job_name in job_names:
        if job_name.startswith("Gears_"):
            parts = job_name.split("_")
            if len(parts) >= 2:
                found_suffixes.add(parts[1])

    if not found_suffixes:
        return True, "Could not extract geometry suffixes"

    # Check consistency
    config_suffix = GEOMETRY_SUFFIX

    if config_suffix in found_suffixes:
        if len(found_suffixes) == 1:
            return True, "Perfect match: All jobs are for geometry " + config_suffix
        else:
            return True, "Partial match: Found geometries " + str(
                sorted(list(found_suffixes))
            )
    else:
        return False, {
            "config_suffix": config_suffix,
            "found_suffixes": sorted(list(found_suffixes)),
        }


def run_job_batch(jobs, start_index, batch_size):
    """Run a batch of jobs with staggered submission"""
    end_index = min(start_index + batch_size, len(jobs))
    batch_jobs = jobs[start_index:end_index]

    print(
        "\nBatch "
        + str((start_index // batch_size) + 1)
        + ": Processing jobs "
        + str(start_index + 1)
        + "-"
        + str(end_index)
    )

    # Submit jobs
    submitted_jobs = []
    for i, job in enumerate(batch_jobs):
        job_name = job["name"]
        status = get_comprehensive_job_status(job_name)

        if status == "COMPLETED" and SKIP_EXISTING_ODBS:
            print("  " + job_name + ": SKIPPED (completed)")
            continue

        if status in ["RUNNING", "SUBMITTED"]:
            print("  " + job_name + ": SKIPPED (running)")
            continue

        if status in ["ABORTED", "TERMINATED", "ERROR"]:
            print("  " + job_name + ": RETRYING (was " + status + ")")

        try:
            print("  " + job_name + ": SUBMITTING...")
            mdb.jobs[job_name].submit(consistencyChecking=OFF)
            submitted_jobs.append(job)
            update_job_status_in_database(job_name, "SUBMITTED")

            # Wait between submissions
            if i < len(batch_jobs) - 1 and WAIT_BETWEEN_SUBMISSIONS > 0:
                time.sleep(WAIT_BETWEEN_SUBMISSIONS)

        except Exception as e:
            print("  " + job_name + ": SUBMIT ERROR - " + str(e))
            update_job_status_in_database(job_name, "SUBMIT_ERROR")

    if not submitted_jobs:
        print("  No jobs submitted in this batch")
        return True

    print("  Waiting for " + str(len(submitted_jobs)) + " jobs to complete...")

    # Wait for completion
    for i, job in enumerate(submitted_jobs):
        job_name = job["name"]

        try:
            start_time = time.time()
            mdb.jobs[job_name].waitForCompletion()
            duration = int(time.time() - start_time)

            # Get final status
            final_status = get_comprehensive_job_status(job_name)

            if final_status == "COMPLETED":
                print("  " + job_name + ": COMPLETED (" + str(duration) + "s)")
            else:
                print(
                    "  " + job_name + ": " + final_status + " (" + str(duration) + "s)"
                )

            update_job_status_in_database(job_name, final_status)

        except Exception as e:
            print("  " + job_name + ": WAIT ERROR - " + str(e))
            final_status = get_comprehensive_job_status(job_name)
            update_job_status_in_database(
                job_name, final_status if final_status != "PENDING" else "ERROR"
            )

    return True


def main():
    """Main function to run job submission"""
    print("=== Abaqus Job Submission Manager ===")
    print("Geometry: " + GEOMETRY_SUFFIX + " | Batch size: " + str(BATCH_SIZE))

    # Check geometry consistency
    is_consistent, result = check_geometry_consistency()

    print("\nGeometry Check: " + ("OK - " + result if is_consistent else "ERROR"))
    if not is_consistent:
        print("  Config: " + result["config_suffix"])
        print("  Found: " + str(result["found_suffixes"]))
        print("  Fix GEOMETRY_SUFFIX and restart")
        return

    # Find jobs
    all_job_names, available_jobs = find_all_jobs()

    if not available_jobs:
        print("\nNo jobs available for geometry " + GEOMETRY_SUFFIX)
        print("Make sure you have:")
        print("1. Created models using 03_two_cylinders.py")
        print("2. Set the correct GEOMETRY_SUFFIX")
        return

    # Status overview
    status_counts = get_job_status_summary(available_jobs)

    print("\nJob Status Overview:")
    print("  Total jobs: " + str(len(available_jobs)))
    print("  Completed: " + str(status_counts["COMPLETED"]))
    print("  Aborted/Failed: " + str(status_counts["ABORTED"]))
    print("  Running: " + str(status_counts["RUNNING"]))
    print("  Pending: " + str(status_counts["PENDING"]))
    if status_counts["OTHER"] > 0:
        print("  Other: " + str(status_counts["OTHER"]))

    # Detailed status if requested
    if VERBOSE_STATUS:
        print("\nDetailed Status:")
        for job in available_jobs[:10]:  # Show first 10
            status = get_comprehensive_job_status(job["name"])
            print("  " + job["name"][:50] + ": " + status)
        if len(available_jobs) > 10:
            print("  ... and " + str(len(available_jobs) - 10) + " more")

    # Update database
    update_all_job_statuses(available_jobs)

    # Check if work needed
    jobs_to_process = status_counts["PENDING"] + status_counts["RUNNING"]
    if jobs_to_process == 0:
        print("\nAll jobs completed or failed - nothing to process")
        return

    # Safety check
    if not AUTO_START:
        print("\n*** Set AUTO_START=True to proceed ***")
        return

    # Execution plan
    total_batches = (len(available_jobs) + BATCH_SIZE - 1) // BATCH_SIZE
    if PROCESS_ALL_BATCHES:
        batches_to_process = total_batches
    elif STOP_AFTER_BATCH > 0:
        batches_to_process = min(STOP_AFTER_BATCH, total_batches)
    else:
        batches_to_process = 1

    print("\nExecution Plan:")
    print("  Jobs to process: " + str(jobs_to_process))
    print("  Batches planned: " + str(batches_to_process) + "/" + str(total_batches))

    # Process batches
    for start_index in range(0, len(available_jobs), BATCH_SIZE):
        batch_num = (start_index // BATCH_SIZE) + 1

        if batch_num > batches_to_process:
            break

        # Pause between batches
        if batch_num > 1 and PAUSE_BETWEEN_BATCHES > 0:
            print("\nPausing " + str(PAUSE_BETWEEN_BATCHES) + " seconds...")
            time.sleep(PAUSE_BETWEEN_BATCHES)

        # Run batch
        success = run_job_batch(available_jobs, start_index, BATCH_SIZE)
        if not success:
            break

    # Final summary
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)

    # Final status check
    final_status_counts = get_job_status_summary(available_jobs)
    update_all_job_statuses(available_jobs)

    print("Final Results:")
    print("  Completed: " + str(final_status_counts["COMPLETED"]))
    print("  Failed: " + str(final_status_counts["ABORTED"]))

    total_completed = final_status_counts["COMPLETED"]
    if len(available_jobs) > 0:
        success_rate = int(100.0 * total_completed / len(available_jobs))
        print("  Success rate: " + str(success_rate) + "%")

    # Next steps guidance
    remaining_batches = total_batches - batches_to_process
    if remaining_batches > 0 and not PROCESS_ALL_BATCHES:
        print("\nNext Steps:")
        print("  " + str(remaining_batches) + " batches remaining")
        print("  Set PROCESS_ALL_BATCHES=True or increase STOP_AFTER_BATCH")


if __name__ == "__main__":
    main()
