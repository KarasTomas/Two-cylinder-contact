# Gear Analysis Workflow

## Overview

This workflow sets up and manages Abaqus FEA simulations for gear contact analysis with multiple geometries. The process involves data preparation, configuration management, automated model generation, job execution, and results extraction.

## Directory Structure

```
Gear_Analysis/
├── initial_conditions/
│   ├── Fn_jednotkova_176.txt       # Force data for geometry 176
│   ├── rho_pastorek_176.txt        # Pinion radius data for geometry 176
│   ├── rho_kolo_176.txt            # Gear radius data for geometry 176
│   ├── D_suma_posuv_176.txt        # Displacement data for geometry 176
│   ├── g_alfa_176.txt              # Angular position data for geometry 176
│   ├── Fn_jednotkova_182.txt       # Force data for geometry 182
│   ├── rho_pastorek_182.txt        # Pinion radius data for geometry 182
│   ├── rho_kolo_182.txt            # Gear radius data for geometry 182
│   └── ...                         # Additional geometry files
├── processed_data/
│   ├── Gears_176_downsampled_data.csv  # Contains F, D, and g_alfa data
│   ├── Gears_176_config.txt
│   ├── Gears_182_downsampled_data.csv  # Contains F, D, and g_alfa data
│   ├── Gears_182_config.txt
│   └── ...
├── abaqus_work/
│   ├── 03_two_cylinders.py         # Abaqus analysis script (runs from here)
│   ├── 04_run_jobs.py              # Job submission and monitoring
│   ├── 06_extract_results.py       # Results extraction (Python 2.7)
│   ├── Gears_176.cae               # CAE files saved here
│   ├── Gears_176_partial.cae       # Auto-saved partial files
│   └── (other Abaqus temporary files)
├── results/
│   ├── Gears_176_results/
│   │   ├── Gears_176_model_database.csv
│   │   ├── odb_files/              # Organized ODB, INP, STA files
│   │   └── extracted_data/         # CSV files with results
│   ├── Gears_182_results/
│   │   ├── Gears_182_model_database.csv
│   │   ├── odb_files/
│   │   └── extracted_data/
│   └── ...
├── 01_setup_analysis.py           # Main preparation script
├── 02_sample_tuner.py             # Interactive sampling parameter tuning
├── 05_move_results.py             # File organization (Python 3.12)
├── 07_plot_csv_data.py            # Simple visualization for testing
└── task_workflow.md               # This documentation file
```

## Workflow Steps

### Step 1: Initial Setup

1. Create main `Gear_Analysis/` directory
2. Run `01_setup_analysis.py` to create subdirectories and configuration files
3. Place initial data files in `initial_conditions/` with proper naming convention

### Step 2: Data Preparation (`01_setup_analysis.py`)

1. **File Discovery**: Scan `initial_conditions/` folder for files matching patterns:
   - `Fn_jednotkova_{suffix}.txt` (force data) ⭐ Required
   - `D_suma_posuv_{suffix}.txt` (displacement data) - Optional
   - `rho_pastorek_{suffix}.txt` (pinion radius data) ⭐ Required
   - `rho_kolo_{suffix}.txt` (gear radius data) ⭐ Required
   - `g_alfa_{suffix}.txt` (angular position data) - Optional

2. **Configuration Generation**: Create `Gears_{suffix}_config.txt` for each geometry containing:

   ```
   # Configuration file for gear geometry analysis
   # Generated for geometry: {suffix}
   # Format: parameter_name=value
   # Note: 03_two_cylinders.py runs from abaqus_work/ directory

   elastic_modulus=3200
   poisson_ratio=0.4
   arc_length=6
   model_name_prefix=Gears_{suffix}
   available_load_types=F,D
   selected_load_type=F
   ```

3. **Directory Preparation**: Create results subdirectories for each geometry

### Step 3: Sampling Parameter Tuning (`02_sample_tuner.py`)

**Interactive Process for Each Geometry:**

1. **Configure Parameters**: Edit script variables:

   ```python
   GEOMETRY_SUFFIX = "176"        # Change to target geometry
   TRIM_START = 150               # Points to trim from start
   TRIM_END = 150                 # Points to trim from end
   TARGET_POINTS = 30             # Final number of sampled points
   MIDDLE_REGION_RATIO = 0.5      # Concentration around peak (0.0-1.0)
   PEAK_THRESHOLD = 0.98          # Peak detection threshold (0.0-1.0)
   OUTPUT_TO_CSV = False          # Set True when satisfied
   ```

2. **Preview Mode**: Run with `OUTPUT_TO_CSV = False`
   - Visual plot with 4 subplots: Force, Radii, G_alfa, Displacement
   - Clear feedback on sampling density and distribution
   - Review and adjust parameters as needed

3. **Generate CSV**: Set `OUTPUT_TO_CSV = True` and re-run
   - Creates `Gears_{suffix}_downsampled_data.csv`
   - Shows status overview of all geometries

### Step 4: Abaqus Model Generation (`03_two_cylinders.py`)

**Prerequisites**: Copy `03_two_cylinders.py` to `abaqus_work/` directory

**Configuration**:

```python
GEOMETRY_SUFFIX = "176"          # Target geometry
TESTING_MODE = True              # True = only 2 models, False = all models
AUTO_SAVE_INTERVAL = 5           # Save CAE every N models (0 = disable)
```

**Process**:

1. **Configuration Loading**: Reads `../processed_data/Gears_{suffix}_config.txt`
2. **Data Loading**: Loads `../processed_data/Gears_{suffix}_downsampled_data.csv`
3. **Model Generation**: Creates Abaqus models with parameterized geometry and contact setup
4. **Database Creation**: Updates `Gears_{suffix}_model_database.csv` with job parameters

### Step 5: Job Submission and Monitoring (`04_run_jobs.py`)

**Prerequisites**: Copy `04_run_jobs.py` to `abaqus_work/` directory alongside `03_two_cylinders.py`

**Configuration**:

```python
GEOMETRY_SUFFIX = "176"         # Target geometry (must match model generation)
BATCH_SIZE = 2                  # Number of jobs to run simultaneously
WAIT_BETWEEN_SUBMISSIONS = 30   # Seconds to wait between job submissions
SKIP_EXISTING_ODBS = True       # Skip jobs if ODB already exists

# EXECUTION CONTROL
AUTO_START = False              # Set to True to start immediately
PROCESS_ALL_BATCHES = False     # Set to True to process all batches
STOP_AFTER_BATCH = 3            # Stop after N batches (0 = process all)
PAUSE_BETWEEN_BATCHES = 60      # Seconds to pause between batches
```

**Features**:

1. **Geometry Consistency Check**: Validates that `GEOMETRY_SUFFIX` matches the jobs in the CAE file
2. **Intelligent Job Status Detection**:
   - Only counts jobs as completed if status is "COMPLETED" AND ODB file exists
   - Properly handles aborted/terminated jobs (which still create ODB files)
   - Allows retry of failed jobs
3. **Batch Processing**: Runs jobs in configurable batches to avoid system overload
4. **Database Integration**: Updates `Gears_{suffix}_model_database.csv` with `odb_status` column
5. **Safety Controls**: Requires explicit confirmation via `AUTO_START = True`

**Process**:

1. **Geometry Validation**: Ensures job suffix matches configuration
2. **Status Analysis**: Scans all jobs and categorizes by completion status
3. **Batch Execution**: Submits jobs in batches with configurable delays
4. **Progress Monitoring**: Tracks job completion with detailed status reporting
5. **Database Update**: Updates model database with final job statuses

**Output Example**:

```
============================================================
GEOMETRY CONSISTENCY CHECK
============================================================
OK - Perfect match: All jobs are for geometry 176

============================================================
JOB STATUS OVERVIEW
============================================================
Completed: 15
Aborted/Failed: 3
Running: 0
Pending: 6
Total: 24

============================================================
EXECUTION PLAN
============================================================
Jobs to process: 6
Batch size: 2
Estimated batches: 3
Will process: 3 batch(es)
```

### Step 6: File Organization (`05_move_results.py`)

**Purpose**: Organize completed analysis files into proper directory structure (Python 3.12)

**Configuration**:

```python
GEOMETRY_SUFFIX = "176"         # Change to match your geometry
DRY_RUN = False                 # Set to True to preview moves without actually moving files
FORCE_MOVE = False              # Set to True to move files even if already marked as moved
VERBOSE = True                  # Set to True for detailed output
```

**Process**:

1. **Read Database**: Load `Gears_{suffix}_model_database.csv`
2. **Filter Completed Jobs**: Select rows where `odb_status == 'COMPLETED'`
3. **Create Directory Structure**:

   ```
   results/Gears_{suffix}_results/
   ├── odb_files/
   ├── extracted_data/
   └── Gears_{suffix}_model_database.csv
   ```

4. **Move Files**: For each completed job, move:
   - `{job_name}.odb` → `results/Gears_{suffix}_results/odb_files/`
   - `{job_name}.inp` → `results/Gears_{suffix}_results/odb_files/`
   - `{job_name}.sta` → `results/Gears_{suffix}_results/odb_files/`
5. **Update Database**: Add columns:
   - `odb_moved` (True/False)
   - `result_csv_file` (filename when extraction completes)
   - `result_content` (list of successfully extracted variables)

**Features**:

- **Safety**: Dry run mode and confirmation prompts
- **Error Recovery**: Rollback mechanism for failed moves
- **Database Integration**: Updates move status automatically
- **Validation**: Checks for missing files and reports issues

### Step 7: Results Extraction (`06_extract_results.py`)

**Purpose**: Extract data from ODB files and save as CSV (Python 2.7 - runs in Abaqus)

**Prerequisites**: Copy `06_extract_results.py` to `abaqus_work/` directory

**Configuration**:

```python
GEOMETRY_SUFFIX = "176"
SELECTED_VARIABLES = ['U2', 'CPRESS', 'CFNORM']  # Variables to extract

# Variable configuration with correct Abaqus syntax
VARIABLE_CONFIG = {
    'U2': {
        'variableLabel': 'U', 
        'outputPosition': NODAL, 
        'refinement': (COMPONENT, 'U2'),
        'description': 'Y-displacement'
    },
    'CPRESS': {
        'variableLabel': 'CPRESS', 
        'outputPosition': ELEMENT_NODAL, 
        'description': 'Contact pressure'
    },
    'CFNORM': {
        'variableLabel': 'CNORMF', 
        'outputPosition': ELEMENT_NODAL, 
        'refinement': (COMPONENT, 'CNORMF2'),
        'description': 'Contact normal force'
    }
}

PATH_CONFIG = {
    'pinion': {
        'instance': 'PINION-1',
        'node_set': 'PINION_EDGE',
        'edge_spec': (1, 4, -1)  # (start_edge, end_edge, direction)
    },
    'gear': {
        'instance': 'GEAR-1', 
        'node_set': 'GEAR_EDGE',
        'edge_spec': (1, 2, 1)
    }
}
```

**Process**:

1. **Database-Driven Processing**: Uses `Gears_{suffix}_model_database.csv` to identify files to process
2. **Path Creation**: Creates extraction paths for both pinion and gear edges using edge list method
3. **Variable Extraction**: Extracts selected variables along each path with proper Abaqus syntax
4. **Data Organization**: Creates single CSV per ODB with structure:

   ```
   X_pinion_U2, Y_pinion_U2, X_gear_U2, Y_gear_U2, X_pinion_CPRESS, Y_pinion_CPRESS, ...
   ```

5. **Database Update**: Updates model database with:
   - `result_csv_file`: filename (e.g., `Gears_176_rP_30,51_rG_28,23_F_120,57.csv`)
   - `result_content`: list of successfully extracted variables
6. **Error Handling**: Skips corrupted ODBs, logs failures, continues processing

**Features**:

- **Robust Path Creation**: Uses proven edge list method from your macros
- **Variable Support**: Handles complex variables (U2, CFNORM) with refinement syntax
- **Error Recovery**: Continues processing even if individual jobs fail
- **Comprehensive Logging**: Detailed extraction log with success/failure tracking

**Output Structure**:

```
results/Gears_{suffix}_results/
├── odb_files/
│   ├── Gears_{suffix}_rP_30,51_rG_28,23_F_120,57_res.odb
│   ├── Gears_{suffix}_rP_30,51_rG_28,23_F_120,57_res.inp
│   └── ...
├── extracted_data/
│   ├── Gears_{suffix}_rP_30,51_rG_28,23_F_120,57.csv
│   ├── Gears_{suffix}_rP_30,12_rG_28,84_F_115,23.csv
│   ├── extraction_log.txt
│   └── ...
└── Gears_{suffix}_model_database.csv (updated with extraction status)
```

### Step 8: Testing and Visualization (`07_plot_csv_data.py`)

**Purpose**: Simple script to test and visualize extracted CSV data (Python 3.12)

**Configuration**:

```python
GEOMETRY_SUFFIX = "176"
CSV_FILENAME = "Gears_176_rP_30,51_rG_28,23_F_120,57.csv"  # Change to your actual file
```

**Features**:

- **Quick Validation**: Reads and displays CSV structure
- **Simple Plots**: U2 displacement, contact pressure, contact normal force
- **Data Summary**: Basic statistics and extraction verification
- **Error Handling**: Lists available files if target not found

**Usage**: Perfect for verifying that the extraction pipeline worked correctly before proceeding to advanced analysis.

### Step 9: Analysis and Visualization

**Ready for Python 3.12 Analysis**: With organized CSV files and updated database, the results are ready for:

1. **Parameter Sensitivity Analysis**: Identify critical design parameters
2. **Visualization Suite**: Stress contours, force-displacement curves, contact maps
3. **Comparative Analysis**: Compare results across geometries
4. **Report Generation**: Automated technical reports

## File Naming Conventions

### Input Files (in initial_conditions/)

- Force: `Fn_jednotkova_{suffix}.txt` ⭐
- Displacement: `D_suma_posuv_{suffix}.txt`
- Pinion radius: `rho_pastorek_{suffix}.txt` ⭐
- Gear radius: `rho_kolo_{suffix}.txt` ⭐
- Angular position: `g_alfa_{suffix}.txt`

### Generated Files

- **Config**: `processed_data/Gears_{suffix}_config.txt`
- **Data**: `processed_data/Gears_{suffix}_downsampled_data.csv`
- **Database**: `results/Gears_{suffix}_results/Gears_{suffix}_model_database.csv`
- **CAE**: `abaqus_work/Gears_{suffix}.cae`
- **ODB**: `results/Gears_{suffix}_results/odb_files/Gears_{suffix}_rP_{parameters}_res.odb`
- **Results**: `results/Gears_{suffix}_results/extracted_data/Gears_{suffix}_rP_{parameters}.csv`

### Model Naming Convention

```
Gears_{suffix}_rP_{pinion_r}_rG_{gear_r}_{load_type}_{load_value}_res
```

Example: `Gears_176_rP_30,51_rG_28,23_F_120,57_res`

## Database Schema

### Model Database (`Gears_{suffix}_model_database.csv`)

```csv
odb_name,model_name,pinion_radius,gear_radius,load_type,load_value,g_alfa,displacement_load,odb_status,odb_moved,result_csv_file,result_content
Gears_176_rP_30,51_rG_28,23_F_120,57_res.odb,Gears_176_rP_30.51_rG_28.23_F_120.57,30.512,28.234,F,120.567,1.234,0.0251,COMPLETED,True,Gears_176_rP_30,51_rG_28,23_F_120,57.csv,"['U2','CPRESS','CFNORM']"
```

**Key Columns**:

- **`odb_name`**: Full ODB filename with extension (e.g., `Gears_176_rP_30,51_rG_28,23_F_120,57_res.odb`)
- **`odb_status`**: Job completion status (updated by `04_run_jobs.py`)
- **`odb_moved`**: Whether files have been organized (updated by `05_move_results.py`)
- **`result_csv_file`**: Name of extracted results CSV (updated by `06_extract_results.py`)
- **`result_content`**: List of successfully extracted variables (updated by `06_extract_results.py`)

## Implementation Status

1. ✅ **Directory Structure Creation** (`01_setup_analysis.py`)
2. ✅ **Configuration File Generation** (automatic discovery and setup)
3. ✅ **Interactive Sampling Tuner** (`02_sample_tuner.py`)
4. ✅ **Abaqus Model Generation** (`03_two_cylinders.py`)
5. ✅ **Job Submission & Monitoring** (`04_run_jobs.py`)
6. ✅ **File Organization** (`05_move_results.py`)
7. ✅ **Results Extraction** (`06_extract_results.py`)
8. ✅ **Basic Visualization** (`07_plot_csv_data.py`)
9. ⏳ **Advanced Post-Processing Scripts** (data analysis and visualization)

## Execution Workflow Summary

### Complete Analysis Workflow

1. **Setup**: `01_setup_analysis.py` → `02_sample_tuner.py` (for each geometry)
2. **Model Creation**: `03_two_cylinders.py` (run from `abaqus_work/`)
3. **Job Execution**: `04_run_jobs.py` (run from `abaqus_work/`)
4. **File Organization**: `05_move_results.py` (Python 3.12)
5. **Data Extraction**: `06_extract_results.py` (run from `abaqus_work/`)
6. **Testing**: `07_plot_csv_data.py` (Python 3.12)
7. **Analysis**: Advanced Python 3.12 scripts for visualization and analysis

### Key Features Implemented

**Database-Driven Coordination**:

- All steps coordinate through the model database
- Automatic status tracking and updates
- Resume capability at any step

**Error Resilience**:

- Comprehensive error handling throughout pipeline
- Rollback mechanisms for failed operations
- Detailed logging and status reporting

**Safety Controls**:

- Dry run modes for preview operations
- Confirmation prompts for destructive operations
- Geometry consistency validation

**Batch Processing**:

- Configurable batch sizes for job submission
- Staggered job submission to avoid system overload
- Progress monitoring and status reporting

**Organized Output**:

- Clean separation of files by processing stage
- Standardized naming conventions
- Automated directory structure creation

### Key Benefits

- **Fully Automated Pipeline**: Complete workflow from data preparation to results extraction
- **Database-Driven**: All steps coordinate through model database with comprehensive status tracking
- **Error Resilient**: Comprehensive error handling, logging, and recovery mechanisms
- **Batch Processing**: Efficient handling of large parameter studies with configurable batch sizes
- **Organized Output**: Clean separation of files and results with standardized naming
- **Resume Capability**: Can restart at any step using database information
- **Safety Controls**: Multiple safety mechanisms including dry run modes and confirmation prompts

This workflow provides a robust, scalable, and fully tested system for automated gear contact analysis with comprehensive result tracking and organization.

## Technical Notes

### Python Version Requirements

- **Scripts 01, 02, 05, 07**: Python 3.12 (modern pandas, pathlib, matplotlib)
- **Scripts 03, 04, 06**: Python 2.7 (Abaqus environment)

### Key Implementation Details

**Database Column Handling**:

- Uses `odb_name` column (with .odb extension) as primary key
- Automatic creation of missing columns (`odb_status`, `odb_moved`, etc.)
- Proper handling of job name extraction from ODB names

**File Path Consistency**:

- All scripts handle relative paths correctly
- Consistent directory structure across all processing steps
- Proper handling of special characters in filenames (commas, periods)

**Abaqus Integration**:

- Correct variable extraction syntax for complex variables (U2, CFNORM)
- Proper use of edge list method for path creation
- Handles both nodal and element-nodal output positions

**Error Recovery**:

- Scripts continue processing after individual failures
- Comprehensive status tracking in database
- Detailed logging for troubleshooting

This implementation represents a complete, tested, and production-ready workflow for automated gear contact analysis in Abaqus.

## Notes and Future Improvements

### A) Configuration System Revision Ideas

**Current Issues:**

- Config parameters are hardcoded in `setup_analysis.py`
- No easy way to modify material properties per geometry
- Arc length is fixed for all geometries

**Proposed Improvements:**

1. **Template-Based Config Generation**:

   ```python
   # Allow user to modify template before applying to all geometries
   config_template = {
       "elastic_modulus": 3200,
       "poisson_ratio": 0.4,
       "arc_length": 6,
       # Could be geometry-specific
   }
   ```

2. **Geometry-Specific Overrides**:

   ```
   # In config file:
   elastic_modulus=3200
   poisson_ratio=0.4
   arc_length=6
   # Geometry-specific overrides
   geometry_176_elastic_modulus=3500
   geometry_176_arc_length=8
   ```

3. **Interactive Config Editor**:
   - GUI or CLI tool to modify parameters before analysis
   - Batch parameter updates across multiple geometries
   - Parameter validation and range checking

### B) Advanced Downsampling Options

**Current Limitations:**

- Only peak-focused sampling around force maximum
- Fixed sampling regions (left, middle, right)
- No slope or gradient-based sampling
- Limited visual feedback for optimization

**Proposed Enhancements:**

1. **Multiple Sampling Strategies**:

   ```python
   SAMPLING_STRATEGY = "peak_focused"  # Options: "peak_focused", "slope_based", "gradient_adaptive", "uniform"
   
   # Slope-based sampling
   SLOPE_THRESHOLD = 0.1              # Focus on regions with high slope
   CURVATURE_THRESHOLD = 0.05         # Sample high curvature regions
   
   # Gradient-adaptive sampling
   GRADIENT_WEIGHT = 0.7              # Balance between gradient and uniform sampling
   MIN_POINT_DISTANCE = 2             # Minimum distance between sample points
   ```

2. **Enhanced Region Detection**:
   - **Slope Analysis**: Identify steep gradients in force/displacement curves
   - **Inflection Points**: Detect changes in curve behavior
   - **Multiple Peaks**: Handle data with multiple local maxima
   - **Transition Zones**: Focus on regions between contact states

3. **Visual Improvements for Plot Analysis**:

   ```python
   # Enhanced plotting features
   SHOW_SAMPLING_REGIONS = True       # Highlight left/middle/right regions
   SHOW_GRADIENT_OVERLAY = True       # Color-code by slope magnitude
   SHOW_CURVATURE_ANALYSIS = True     # Mark high curvature points
   PLOT_DERIVATIVES = True            # Show 1st and 2nd derivatives
   INTERACTIVE_SELECTION = True       # Allow manual point selection
   ```

4. **Adaptive Sampling Algorithms**:
   - **Error-Based**: Sample more densely where interpolation error is high
   - **Feature-Preserving**: Ensure critical features (peaks, valleys) are captured
   - **Physics-Informed**: Consider contact mechanics for intelligent sampling

### C) Model Database Integration for Complementary Sampling

**Current State:**

- Model database tracks completed simulations
- No integration with sampling decisions
- Manual coordination between existing and new models

**Proposed Integration System:**

1. **Gap Analysis Tool**:

   ```python
   # New script: gap_analyzer.py
   def analyze_parameter_coverage(geometry_suffix):
       """Analyze existing model coverage and identify gaps"""
       database = load_model_database(geometry_suffix)
       current_data = load_downsampled_data(geometry_suffix)
       
       # Identify parameter space gaps
       gaps = find_coverage_gaps(database, parameter_bounds)
       
       # Suggest complementary sampling points
       suggested_points = generate_complementary_samples(gaps, target_density)
       
       return gaps, suggested_points
   ```

2. **Smart Resampling Strategy**:

   ```python
   # In sample_tuner.py
   COMPLEMENT_EXISTING = True         # Consider existing models when sampling
   TARGET_COVERAGE = 0.85            # Desired parameter space coverage
   AVOID_REDUNDANCY_DISTANCE = 0.1   # Minimum distance from existing points
   ```

3. **Incremental Model Generation**:
   - **Database-Aware Sampling**: Skip parameter combinations already simulated
   - **Coverage Optimization**: Focus new points where coverage is sparse
   - **Result-Informed Sampling**: Sample more densely in regions with interesting results

4. **Multi-Stage Workflow**:

   ```
   Stage 1: Initial coarse sampling (20-30 points)
   Stage 2: Run analysis and evaluate results
   Stage 3: Adaptive refinement based on:
            - High stress gradients
            - Convergence issues
            - Interesting physical phenomena
   Stage 4: Final validation sampling
   ```

5. **Cross-Geometry Learning**:

   ```python
   # Learn from other geometries
   def suggest_sampling_from_similar_geometries():
       """Use sampling patterns from similar geometries"""
       similar_geometries = find_similar_geometries(current_geometry)
       successful_patterns = extract_sampling_patterns(similar_geometries)
       return adapt_patterns_to_current_geometry(successful_patterns)
   ```

### D) Workflow Enhancements

**Batch Processing**:

- Process multiple geometries in sequence automatically
- Queue management for large geometry sets
- Parallel processing capabilities

**Resume Functionality**:

- Check existing models before creating new ones
- Skip already completed parameter combinations
- Robust restart after interruption

**Results Management**:

- Automatic ODB file organization
- Result extraction and consolidation
- Parameter sensitivity analysis tools

### E) Analysis Improvements

**Adaptive Meshing**:

- Mesh refinement based on contact pressure
- Automatic convergence studies
- Mesh quality metrics and reporting

**Advanced Contact Modeling**:

- Friction coefficient parametrization
- Surface roughness effects
- Wear prediction capabilities

**Optimization Integration**:

- Parameter optimization workflows
- Design of experiments (DOE) setup
- Multi-objective optimization

### F) User Experience

**Documentation**:

- Video tutorials for workflow steps
- Troubleshooting guide with common issues
- Best practices for parameter selection

**Validation Tools**:

- Result verification against analytical solutions
- Benchmark test cases
- Quality assurance metrics

**Integration**:

- CAD software integration for geometry import
- Results export to common analysis tools
- Report generation automation

### G) Technical Debt

**Code Organization**:

- Modularize `two_cylinders.py` into functions
- Create shared utility functions between scripts
- Implement proper logging system

**Testing**:

- Unit tests for core functions
- Integration tests for complete workflow
- Automated testing with sample data

**Performance**:

- Memory usage optimization for large datasets
- Parallel model generation
- Database indexing for large result sets

## Priority Implementation Roadmap

### Phase 1: Enhanced Sampling (High Priority)

1. **Slope-based sampling algorithm**
2. **Visual plot improvements** (region highlighting, gradient overlay)
3. **Gap analysis tool** for existing model integration
4. **Multiple sampling strategy options**

### Phase 2: Smart Integration (Medium Priority)

1. **Database-aware sampling** to avoid redundancy
2. **Incremental model generation** workflow
3. **Cross-geometry learning** system
4. **Result-informed adaptive sampling**

### Phase 3: Advanced Features (Lower Priority)

1. **Interactive point selection** in plots
2. **Physics-informed sampling** algorithms
3. **Multi-stage sampling** workflow automation
4. **Advanced visualization** with derivatives and curvature analysis

This roadmap prioritizes the most impactful improvements that directly address current workflow limitations while building toward a more sophisticated and automated system.
