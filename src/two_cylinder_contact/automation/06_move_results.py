"""Tools for moving the simulation results to the appropriate directory."""

import shutil
import sys
from pathlib import Path

import pandas as pd

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
GEOMETRY_SUFFIX = "179"  # Change this to match your geometry
DRY_RUN = True  # Set to True to preview moves without actually moving files
FORCE_MOVE = False  # Set to True to move files even if already marked as moved
VERBOSE = True  # Set to True for detailed output

# Paths
ABAQUS_WORK_DIR = "abaqus_work"
RESULTS_BASE_DIR = "results"
# ==============================================================================


def get_paths():
    """Get all relevant paths for the specified geometry, relative to the cwd (src)"""
    paths = {
        "abaqus_work": Path("./abaqus_work"),
        "results_base": Path("./results"),
        "geometry_results": Path(f"./results/gear_{GEOMETRY_SUFFIX}_results"),
        "odb_files": Path(f"./results/gear_{GEOMETRY_SUFFIX}_results/odb_files"),
        "extracted_data": Path(
            f"./results/gear_{GEOMETRY_SUFFIX}_results/extracted_data"
        ),
        "database": Path(
            f"./results/gear_{GEOMETRY_SUFFIX}_results/gear_{GEOMETRY_SUFFIX}_model_database.csv"
        ),
    }
    return paths


def validate_environment():
    """Validate that required directories and files exist"""
    paths = get_paths()
    issues = []

    # Check if abaqus_work directory exists
    if not paths["abaqus_work"].exists():
        issues.append(f"Abaqus work directory not found: {paths['abaqus_work']}")

    # Check if results base directory exists
    if not paths["results_base"].exists():
        issues.append(f"Results base directory not found: {paths['results_base']}")

    # Check if geometry results directory exists
    if not paths["geometry_results"].exists():
        issues.append(
            f"Geometry results directory not found: {paths['geometry_results']}"
        )

    # Check if database file exists
    if not paths["database"].exists():
        issues.append(f"Model database not found: {paths['database']}")

    return issues


def read_database():
    """Read the model database and return jobs to process"""
    paths = get_paths()

    try:
        df = pd.read_csv(paths["database"])
        print(f"✓ Read database: {len(df)} total jobs found")

        # Check if odb_status column exists
        if "odb_status" not in df.columns:
            print("WARNING: 'odb_status' column not found in database")
            print("Please run run_jobs.py first to populate job statuses")
            return None

        # Add odb_moved column if it doesn't exist
        if "odb_moved" not in df.columns:
            df["odb_moved"] = False
            print("✓ Added 'odb_moved' column to database")

        # Add result_csv_file column if it doesn't exist
        if "result_csv_file" not in df.columns:
            df["result_csv_file"] = ""
            print("✓ Added 'result_csv_file' column to database")

        # Add result_content column if it doesn't exist
        if "result_content" not in df.columns:
            df["result_content"] = ""
            print("✓ Added 'result_content' column to database")

        return df

    except Exception as e:
        print(f"ERROR reading database: {e}")
        return None


def find_files_to_move(df):
    """Find files that need to be moved based on database"""
    paths = get_paths()

    # Filter for completed jobs
    completed_jobs = df[df["odb_status"] == "COMPLETED"].copy()
    print(f"✓ Found {len(completed_jobs)} completed jobs")

    # Filter for jobs not yet moved (unless FORCE_MOVE is True)
    if not FORCE_MOVE:
        jobs_to_move = completed_jobs[completed_jobs["odb_moved"] != True].copy()
        already_moved = len(completed_jobs) - len(jobs_to_move)
        if already_moved > 0:
            print(
                f"  {already_moved} jobs already moved (use FORCE_MOVE=True to re-move)"
            )
    else:
        jobs_to_move = completed_jobs.copy()
        print("  FORCE_MOVE enabled - will move all completed jobs")

    print(f"✓ {len(jobs_to_move)} jobs selected for moving")

    # Find actual files for each job
    files_to_move = []
    missing_files = []

    for _, job in jobs_to_move.iterrows():
        # Use odb_name and remove .odb extension to get job_name
        odb_name = job["odb_name"]
        job_name = (
            odb_name.replace(".odb", "") if odb_name.endswith(".odb") else odb_name
        )

        # Expected file extensions
        file_extensions = [".odb", ".inp", ".sta"]
        job_files = {"job_name": job_name, "odb_name": odb_name, "files": {}}

        for ext in file_extensions:
            source_file = paths["abaqus_work"] / f"{job_name}{ext}"

            if source_file.exists():
                job_files["files"][ext] = source_file
                if VERBOSE:
                    print(f"    Found: {source_file.name}")
            else:
                missing_files.append(f"{job_name}{ext}")
                if VERBOSE:
                    print(f"    Missing: {source_file.name}")

        # Only include jobs that have at least the ODB file
        if ".odb" in job_files["files"]:
            files_to_move.append(job_files)
        else:
            print(f"  WARNING: No ODB file found for {job_name} - skipping")

    if missing_files:
        print(f"\nWARNING: {len(missing_files)} files not found:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    return files_to_move, jobs_to_move


def create_directories():
    """Create necessary directory structure"""
    paths = get_paths()

    directories_to_create = [
        paths["geometry_results"],
        paths["odb_files"],
        paths["extracted_data"],
    ]

    for directory in directories_to_create:
        if not directory.exists():
            if not DRY_RUN:
                directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        else:
            if VERBOSE:
                print(f"  Directory exists: {directory}")


def move_files(files_to_move):
    """Move files to organized directory structure"""
    paths = get_paths()
    moved_jobs = []
    failed_moves = []

    print(f"\n{'=' * 60}")
    print("MOVING FILES")
    print(f"{'=' * 60}")

    for job_info in files_to_move:
        job_name = job_info["job_name"]
        files = job_info["files"]

        print(f"\nProcessing job: {job_name}")

        job_success = True
        moved_files = []

        for ext, source_file in files.items():
            destination_file = paths["odb_files"] / source_file.name

            try:
                if DRY_RUN:
                    print(
                        f"  DRY RUN: Would move {source_file.name} -> {destination_file}"
                    )
                else:
                    # Check if destination already exists
                    if destination_file.exists():
                        print(
                            f"  WARNING: Destination exists, overwriting: {destination_file.name}"
                        )

                    shutil.move(str(source_file), str(destination_file))
                    print(f"  ✓ Moved: {source_file.name}")
                    moved_files.append(destination_file.name)

            except Exception as e:
                print(f"  ✗ FAILED to move {source_file.name}: {e}")
                job_success = False
                break

        if job_success:
            moved_jobs.append(job_name)
        else:
            failed_moves.append(job_name)
            # Try to rollback any files that were moved for this job
            if not DRY_RUN:
                for moved_file in moved_files:
                    try:
                        rollback_source = paths["odb_files"] / moved_file
                        rollback_dest = paths["abaqus_work"] / moved_file
                        if rollback_source.exists():
                            shutil.move(str(rollback_source), str(rollback_dest))
                            print(f"  ↺ Rolled back: {moved_file}")
                    except Exception as rollback_error:
                        print(f"  ⚠ Failed to rollback {moved_file}: {rollback_error}")

    return moved_jobs, failed_moves


def update_database(df, moved_jobs, failed_moves):
    """Update the database with move status"""
    paths = get_paths()

    print(f"\n{'=' * 60}")
    print("UPDATING DATABASE")
    print(f"{'=' * 60}")

    # Update moved jobs - use odb_name for matching
    for job_name in moved_jobs:
        # Find the row by matching job_name with odb_name (without .odb)
        mask = df["odb_name"].str.replace(".odb", "") == job_name
        if not DRY_RUN:
            df.loc[mask, "odb_moved"] = True
        print(f"  ✓ Updated: {job_name} -> odb_moved = True")

    # Mark failed moves
    for job_name in failed_moves:
        mask = df["odb_name"].str.replace(".odb", "") == job_name
        if not DRY_RUN:
            df.loc[mask, "odb_moved"] = False
        print(f"  ✗ Failed: {job_name} -> odb_moved = False")

    # Save updated database
    if not DRY_RUN:
        try:
            df.to_csv(paths["database"], index=False)
            print(f"✓ Database updated: {paths['database']}")
        except Exception as e:
            print(f"✗ Failed to update database: {e}")
            return False
    else:
        print("  DRY RUN: Database would be updated")

    return True


def generate_summary(df, moved_jobs, failed_moves):
    """Generate and display summary report"""
    paths = get_paths()

    print(f"\n{'=' * 60}")
    print("SUMMARY REPORT")
    print(f"{'=' * 60}")

    # Count jobs by status
    completed_jobs = len(df[df["odb_status"] == "COMPLETED"])
    moved_count = len(df[df["odb_moved"] == True])
    pending_extraction = len(
        df[(df["odb_moved"] == True) & (df["result_csv_file"] == "")]
    )

    print(f"Geometry: {GEOMETRY_SUFFIX}")
    print(f"Database: {paths['database']}")
    print("")
    print("Job Status Summary:")
    print(f"  Total jobs in database: {len(df)}")
    print(f"  Completed jobs: {completed_jobs}")
    print(f"  Files moved: {moved_count}")
    print(f"  Ready for extraction: {pending_extraction}")
    print("")
    print("This Session:")
    print(f"  Successfully moved: {len(moved_jobs)}")
    print(f"  Failed moves: {len(failed_moves)}")

    if moved_jobs:
        print("\nMoved Jobs:")
        for job in moved_jobs[:5]:  # Show first 5
            print(f"  ✓ {job}")
        if len(moved_jobs) > 5:
            print(f"  ... and {len(moved_jobs) - 5} more")

    if failed_moves:
        print("\nFailed Moves:")
        for job in failed_moves:
            print(f"  ✗ {job}")

    print("\nNext Steps:")
    if pending_extraction > 0:
        print(f"  1. Run extract_results.py to process {pending_extraction} ODB files")
        print(f"  2. Results will be saved to: {paths['extracted_data']}")
    else:
        print("  All files have been processed or no completed jobs found")

    print("\nFile Locations:")
    print(f"  ODB files: {paths['odb_files']}")
    print(f"  Extracted data: {paths['extracted_data']}")


def main():
    """Main function to orchestrate the file moving process"""
    print("=" * 60)
    print("ABAQUS RESULTS FILE ORGANIZER")
    print("=" * 60)
    print(f"Geometry: {GEOMETRY_SUFFIX}")
    print(f"Dry run: {DRY_RUN}")
    print(f"Force move: {FORCE_MOVE}")

    # Validate environment
    print(f"\n{'=' * 60}")
    print("ENVIRONMENT VALIDATION")
    print(f"{'=' * 60}")

    issues = validate_environment()
    if issues:
        print("✗ Environment validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease resolve these issues and run again.")
        return 1

    print("✓ Environment validation passed")

    # Read database
    print(f"\n{'=' * 60}")
    print("DATABASE ANALYSIS")
    print(f"{'=' * 60}")

    df = read_database()
    if df is None:
        return 1

    # Find files to move
    files_to_move, jobs_to_move = find_files_to_move(df)

    if not files_to_move:
        print("\nNo files to move. All completed jobs have already been moved.")
        generate_summary(df, [], [])
        return 0

    # Create directories
    print(f"\n{'=' * 60}")
    print("DIRECTORY SETUP")
    print(f"{'=' * 60}")

    create_directories()

    # Confirm operation (unless dry run)
    if not DRY_RUN:
        print(f"\n{'=' * 60}")
        print("CONFIRMATION")
        print(f"{'=' * 60}")
        print(
            f"Ready to move {len(files_to_move)} jobs ({sum(len(job['files']) for job in files_to_move)} files)"
        )
        print(f"From: {get_paths()['abaqus_work']}")
        print(f"To: {get_paths()['odb_files']}")

        response = input("\nProceed with file moves? (y/N): ").strip().lower()
        if response != "y":
            print("Operation cancelled by user")
            return 0

    # Move files
    moved_jobs, failed_moves = move_files(files_to_move)

    # Update database
    if moved_jobs or failed_moves:
        success = update_database(df, moved_jobs, failed_moves)
        if not success:
            print("WARNING: Database update failed")

    # Generate summary
    generate_summary(df, moved_jobs, failed_moves)

    # Return appropriate exit code
    if failed_moves:
        print(f"\n⚠ Completed with {len(failed_moves)} failures")
        return 1
    else:
        print("\n✓ All operations completed successfully")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
