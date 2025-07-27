# Tools for modeling the two-cylinder contact problem in a Abaqus CAE 2019 environment.
# Uses Python 2.7
# All necessary imports
import csv
import math
import os

from assembly import *
from connectorBehavior import *
from interaction import *
from job import *
from load import *
from material import *
from mesh import *
from optimization import *
from part import *
from section import *
from sketch import *
from step import *
from visualization import *

# ==============================================================================
# USER CONFIGURATION - SPECIFY GEOMETRY TO ANALYZE
# ==============================================================================
GEOMETRY_SUFFIX = "179"  # Change this to the geometry you want to analyze
TESTING_MODE = True  # Set to True to create only first 6 models for testing
AUTO_SAVE_INTERVAL = 5  # Save CAE file every N models (0 to disable)
# ==============================================================================


def load_config_file(geometry_suffix):
    """Load configuration parameters for a geometry from abaqus_config.csv (Python 2.7 compatible, no pandas)."""
    config_file = "../processed_data/abaqus_config.csv"
    if not os.path.exists(config_file):
        print("Error: Config file not found: " + config_file)
        return None

    config = {}
    with open(config_file, "r") as f:
        reader = csv.DictReader(f)
        found = False
        for row in reader:
            # geometry_id may be like "gear_176", so match suffix
            if row.get("geometry_id", "").endswith(str(geometry_suffix)):
                config = row
                found = True
                break
        if not found:
            print(
                "Error: Geometry with suffix '{}' not found in config file.".format(
                    geometry_suffix
                )
            )
            return None

    print("Loaded configuration for geometry " + geometry_suffix)
    return config


def load_downsampled_data(geometry_suffix, selected_load_type):
    """Load downsampled data from CSV file"""
    csv_file = "../processed_data/gears_" + geometry_suffix + "_downsampled_data.csv"

    if not os.path.exists(csv_file):
        print("Error: Data file not found: " + csv_file)
        return None, None, None

    pinion_radius = []
    gear_radius = []
    load_vector = []

    with open(csv_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read header

        # Find the correct load column based on load type
        if selected_load_type == "F":
            load_column_name = "force_load"
        elif selected_load_type == "D":
            load_column_name = "displacement_load"
        else:
            print("Error: Unknown load type: " + selected_load_type)
            return None, None, None

        # Find column indices
        try:
            pinion_col = header.index("pinion_radius")
            gear_col = header.index("gear_radius")
            load_col = header.index(load_column_name)
        except ValueError as e:
            print("Error: Required column not found in CSV: " + str(e))
            return None, None, None

        # Read data
        for row in reader:
            pinion_radius.append(float(row[pinion_col]))
            gear_radius.append(float(row[gear_col]))
            load_vector.append(float(row[load_col]))

    print(
        "Loaded "
        + str(len(pinion_radius))
        + " data points for load type: "
        + selected_load_type
    )
    return pinion_radius, gear_radius, load_vector


def setup_results_directory(config):
    """Create results directory if it doesn't exist"""
    results_dir = "../results/" + config["geometry_id"] + "_results"
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print("Created results directory: " + results_dir)
        except:
            print("Warning: Could not create results directory")
    return results_dir


def main():
    """Main function"""
    print("=== Abaqus Gear Analysis ===")
    print("Geometry: " + GEOMETRY_SUFFIX)

    if TESTING_MODE:
        print("TESTING MODE: Will create only 2 models")

    # Load configuration
    config = load_config_file(GEOMETRY_SUFFIX)
    if not config:
        return

    # Extract configuration parameters
    model_name_prefix = config["geometry_id"]
    selected_load_type = config["selected_load_type"]
    elastic_modulus = float(config["elastic_modulus"])
    nu = float(config["poisson_ratio"])
    arc_length = float(config["arc_length"])

    # Constants
    pi = math.pi

    print("Configuration loaded:")
    print("  Model prefix: " + model_name_prefix)
    print("  Load type: " + selected_load_type)
    print("  Elastic modulus: " + str(elastic_modulus))
    print("  Poisson ratio: " + str(nu))

    # Load downsampled data
    pinion_radius, gear_radius, load_vector = load_downsampled_data(
        GEOMETRY_SUFFIX, selected_load_type
    )
    if not pinion_radius:
        return

    # Setup results directory
    results_dir = setup_results_directory(config)

    # Create model database CSV
    model_dict = {
        "model_name": [],
        "pinion_radius": [],
        "gear_radius": [],
        "load_type": [],
        "load_magnitude": [],
        "odb_name": [],
    }

    fieldnames = [
        "model_name",
        "pinion_radius",
        "gear_radius",
        "load_type",
        "load_magnitude",
        "odb_name",
    ]

    # Create CSV file path - use results directory
    database_csv_path = results_dir + "/" + model_name_prefix + "_model_database.csv"

    # Create CSV file and write header
    with open(database_csv_path, "wb") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print("Created model database: " + database_csv_path)

    # Determine total models to process
    total_models = len(pinion_radius)
    if TESTING_MODE:
        total_models = min(6, total_models)

    print("Processing " + str(total_models) + " models...")

    # Model counter
    models_created = 0

    for i, (pinion_r, gear_r, load) in enumerate(
        zip(pinion_radius, gear_radius, load_vector)
    ):
        # Stop if in testing mode and reached limit
        if TESTING_MODE and models_created >= 6:
            print("Testing mode: Stopping after 6 models")
            break

        models_created += 1
        print("\n" + "=" * 50)
        print("Processing model " + str(models_created) + "/" + str(total_models))
        print("  Pinion radius: " + str(pinion_r))
        print("  Gear radius: " + str(gear_r))
        print("  Load: " + str(load))

        # Create model name with rounding for cleaner names
        pinion_r_rounded = round(pinion_r, 2)
        gear_r_rounded = round(gear_r, 2)
        load_rounded = round(load, 2)

        name = (
            str(pinion_r_rounded)
            + "_rG_"
            + str(gear_r_rounded)
            + "_"
            + selected_load_type
            + "_"
            + str(load_rounded)
        )
        name = name.replace(".", ",")
        model_name = str(model_name_prefix + "_rP_" + name)

        print("  Model name: " + model_name)

        # Write down model parameters (use original values, not rounded)
        model_dict["model_name"].append(model_name)
        model_dict["pinion_radius"].append(pinion_r)
        model_dict["gear_radius"].append(gear_r)
        model_dict["load_type"].append(selected_load_type)
        model_dict["load_magnitude"].append(load)
        odb_name = model_name + "_res"

        # new model
        mdb.Model(modelType=STANDARD_EXPLICIT, name=model_name)

        # dimensions
        gear_theta = (360 * arc_length) / (2 * pi * gear_r)
        pinion_theta = (360 * arc_length) / (2 * pi * pinion_r)

        # new part
        mdb.models[model_name].ConstrainedSketch(name="__profile__", sheetSize=200.0)
        mdb.models[model_name].sketches["__profile__"].ArcByCenterEnds(
            center=(0.0, -pinion_r),
            direction=CLOCKWISE,
            point1=(-pinion_r, -pinion_r),
            point2=(pinion_r, -pinion_r),
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(-pinion_r, -pinion_r), point2=(pinion_r, -pinion_r)
        )
        mdb.models[model_name].sketches["__profile__"].HorizontalConstraint(
            addUndoState=False,
            entity=mdb.models[model_name].sketches["__profile__"].geometry[3],
        )
        mdb.models[model_name].Part(
            dimensionality=TWO_D_PLANAR, name="Pinion", type=DEFORMABLE_BODY
        )
        mdb.models[model_name].parts["Pinion"].BaseShell(
            sketch=mdb.models[model_name].sketches["__profile__"]
        )
        del mdb.models[model_name].sketches["__profile__"]

        # first partition
        mdb.models[model_name].ConstrainedSketch(
            gridSpacing=2.19,
            name="__profile__",
            sheetSize=87.74,
            transform=mdb.models[model_name]
            .parts["Pinion"]
            .MakeSketchTransform(
                sketchPlane=mdb.models[model_name].parts["Pinion"].faces[0],
                sketchPlaneSide=SIDE1,
                sketchOrientation=RIGHT,
                origin=(0.0, 0.0, 0.0),
            ),
        )
        mdb.models[model_name].parts["Pinion"].projectReferencesOntoSketch(
            filter=COPLANAR_EDGES, sketch=mdb.models[model_name].sketches["__profile__"]
        )
        mdb.models[model_name].sketches["__profile__"].ArcByCenterEnds(
            center=(0.0, -pinion_r),
            direction=COUNTERCLOCKWISE,
            point1=(pinion_r - 2, -pinion_r),
            point2=(-pinion_r + 2, -pinion_r),
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[3],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[2],
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[4],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[2],
        )
        mdb.models[model_name].sketches["__profile__"].RadialDimension(
            curve=mdb.models[model_name].sketches["__profile__"].geometry[4],
            radius=pinion_r - 2,
            textPoint=(11.9338283538818, 9.0472740168457),
        )
        mdb.models[model_name].parts["Pinion"].PartitionFaceBySketch(
            faces=mdb.models[model_name]
            .parts["Pinion"]
            .faces.getSequenceFromMask(
                ("[#1 ]",),
            ),
            sketch=mdb.models[model_name].sketches["__profile__"],
        )
        del mdb.models[model_name].sketches["__profile__"]

        # Second partition
        mdb.models[model_name].ConstrainedSketch(
            gridSpacing=3.35,
            name="__profile__",
            sheetSize=134.16,
            transform=mdb.models[model_name]
            .parts["Pinion"]
            .MakeSketchTransform(
                sketchPlane=mdb.models[model_name].parts["Pinion"].faces[0],
                sketchPlaneSide=SIDE1,
                sketchOrientation=RIGHT,
                origin=(0.0, 0.0, 0.0),
            ),
        )
        mdb.models[model_name].parts["Pinion"].projectReferencesOntoSketch(
            filter=COPLANAR_EDGES, sketch=mdb.models[model_name].sketches["__profile__"]
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(0.0, -pinion_r), point2=(0.0, 0.0)
        )
        mdb.models[model_name].sketches["__profile__"].VerticalConstraint(
            addUndoState=False,
            entity=mdb.models[model_name].sketches["__profile__"].geometry[8],
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[5],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[4],
        )
        mdb.models[model_name].sketches["__profile__"].EqualDistanceConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[3],
            entity2=mdb.models[model_name].sketches["__profile__"].vertices[4],
            midpoint=mdb.models[model_name].sketches["__profile__"].vertices[5],
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(0.0, -pinion_r), point2=(pinion_r / 5, 0.0)
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[6],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[4],
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(0.0, -pinion_r), point2=(-pinion_r / 5, 0.0)
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[7],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[4],
        )
        mdb.models[model_name].sketches["__profile__"].setAsConstruction(
            objectList=(mdb.models[model_name].sketches["__profile__"].geometry[8],)
        )
        mdb.models[model_name].sketches["__profile__"].SymmetryConstraint(
            entity1=mdb.models[model_name].sketches["__profile__"].geometry[10],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[9],
            symmetryAxis=mdb.models[model_name].sketches["__profile__"].geometry[8],
        )
        mdb.models[model_name].sketches["__profile__"].AngularDimension(
            line1=mdb.models[model_name].sketches["__profile__"].geometry[10],
            line2=mdb.models[model_name].sketches["__profile__"].geometry[9],
            textPoint=(3.14194297790527, 3.94467803991699),
            value=pinion_theta,
        )
        mdb.models[model_name].parts["Pinion"].PartitionFaceBySketch(
            faces=mdb.models[model_name]
            .parts["Pinion"]
            .faces.getSequenceFromMask(
                ("[#1 ]",),
            ),
            sketch=mdb.models[model_name].sketches["__profile__"],
        )
        del mdb.models[model_name].sketches["__profile__"]

        # next part
        mdb.models[model_name].ConstrainedSketch(name="__profile__", sheetSize=200.0)
        mdb.models[model_name].sketches["__profile__"].ArcByCenterEnds(
            center=(0.0, gear_r),
            direction=CLOCKWISE,
            point1=(gear_r, gear_r),
            point2=(-gear_r, gear_r),
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(gear_r, gear_r), point2=(-gear_r, gear_r)
        )
        mdb.models[model_name].sketches["__profile__"].HorizontalConstraint(
            addUndoState=False,
            entity=mdb.models[model_name].sketches["__profile__"].geometry[3],
        )
        mdb.models[model_name].Part(
            dimensionality=TWO_D_PLANAR, name="Gear", type=DEFORMABLE_BODY
        )
        mdb.models[model_name].parts["Gear"].BaseShell(
            sketch=mdb.models[model_name].sketches["__profile__"]
        )
        del mdb.models[model_name].sketches["__profile__"]

        # first partition
        mdb.models[model_name].ConstrainedSketch(
            gridSpacing=2.19,
            name="__profile__",
            sheetSize=87.74,
            transform=mdb.models[model_name]
            .parts["Gear"]
            .MakeSketchTransform(
                sketchPlane=mdb.models[model_name].parts["Gear"].faces[0],
                sketchPlaneSide=SIDE1,
                sketchOrientation=RIGHT,
                origin=(0.0, 0.0, 0.0),
            ),
        )
        mdb.models[model_name].parts["Gear"].projectReferencesOntoSketch(
            filter=COPLANAR_EDGES, sketch=mdb.models[model_name].sketches["__profile__"]
        )
        mdb.models[model_name].sketches["__profile__"].ArcByCenterEnds(
            center=(0.0, gear_r),
            direction=COUNTERCLOCKWISE,
            point1=(-gear_r + 2, gear_r),
            point2=(gear_r - 2, gear_r),
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[3],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[2],
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[4],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[2],
        )
        mdb.models[model_name].sketches["__profile__"].RadialDimension(
            curve=mdb.models[model_name].sketches["__profile__"].geometry[4],
            radius=gear_r - 2,
            textPoint=(11.9338283538818, 9.0472740168457),
        )
        mdb.models[model_name].parts["Gear"].PartitionFaceBySketch(
            faces=mdb.models[model_name]
            .parts["Gear"]
            .faces.getSequenceFromMask(
                ("[#1 ]",),
            ),
            sketch=mdb.models[model_name].sketches["__profile__"],
        )
        del mdb.models[model_name].sketches["__profile__"]

        # Second partition
        mdb.models[model_name].ConstrainedSketch(
            gridSpacing=3.35,
            name="__profile__",
            sheetSize=134.16,
            transform=mdb.models[model_name]
            .parts["Gear"]
            .MakeSketchTransform(
                sketchPlane=mdb.models[model_name].parts["Gear"].faces[0],
                sketchPlaneSide=SIDE1,
                sketchOrientation=RIGHT,
                origin=(0.0, 0.0, 0.0),
            ),
        )
        mdb.models[model_name].parts["Gear"].projectReferencesOntoSketch(
            filter=COPLANAR_EDGES, sketch=mdb.models[model_name].sketches["__profile__"]
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(0.0, gear_r), point2=(0.0, 0.0)
        )
        mdb.models[model_name].sketches["__profile__"].VerticalConstraint(
            addUndoState=False,
            entity=mdb.models[model_name].sketches["__profile__"].geometry[8],
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[5],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[4],
        )
        mdb.models[model_name].sketches["__profile__"].EqualDistanceConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[3],
            entity2=mdb.models[model_name].sketches["__profile__"].vertices[4],
            midpoint=mdb.models[model_name].sketches["__profile__"].vertices[5],
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(0.0, gear_r), point2=(gear_r / 5, 0.0)
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[6],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[4],
        )
        mdb.models[model_name].sketches["__profile__"].Line(
            point1=(0.0, gear_r), point2=(-gear_r / 5, 0.0)
        )
        mdb.models[model_name].sketches["__profile__"].CoincidentConstraint(
            addUndoState=False,
            entity1=mdb.models[model_name].sketches["__profile__"].vertices[7],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[4],
        )
        mdb.models[model_name].sketches["__profile__"].setAsConstruction(
            objectList=(mdb.models[model_name].sketches["__profile__"].geometry[8],)
        )
        mdb.models[model_name].sketches["__profile__"].SymmetryConstraint(
            entity1=mdb.models[model_name].sketches["__profile__"].geometry[10],
            entity2=mdb.models[model_name].sketches["__profile__"].geometry[9],
            symmetryAxis=mdb.models[model_name].sketches["__profile__"].geometry[8],
        )
        mdb.models[model_name].sketches["__profile__"].AngularDimension(
            line1=mdb.models[model_name].sketches["__profile__"].geometry[10],
            line2=mdb.models[model_name].sketches["__profile__"].geometry[9],
            textPoint=(-3.14194297790527, -3.94467803991699),
            value=gear_theta,
        )
        mdb.models[model_name].parts["Gear"].PartitionFaceBySketch(
            faces=mdb.models[model_name]
            .parts["Gear"]
            .faces.getSequenceFromMask(
                ("[#1 ]",),
            ),
            sketch=mdb.models[model_name].sketches["__profile__"],
        )
        del mdb.models[model_name].sketches["__profile__"]

        # create material and section
        mdb.models[model_name].Material(name="plexiglass")
        mdb.models[model_name].materials["plexiglass"].Elastic(
            table=((elastic_modulus, nu),)
        )
        mdb.models[model_name].HomogeneousSolidSection(
            material="plexiglass", name="Section-1", thickness=None
        )
        mdb.models[model_name].parts["Gear"].Set(
            faces=mdb.models[model_name]
            .parts["Gear"]
            .faces.getSequenceFromMask(
                ("[#f ]",),
            ),
            name="gear_set",
        )
        mdb.models[model_name].parts["Gear"].SectionAssignment(
            offset=0.0,
            offsetField="",
            offsetType=MIDDLE_SURFACE,
            region=mdb.models[model_name].parts["Gear"].sets["gear_set"],
            sectionName="Section-1",
            thicknessAssignment=FROM_SECTION,
        )
        mdb.models[model_name].parts["Pinion"].Set(
            faces=mdb.models[model_name]
            .parts["Pinion"]
            .faces.getSequenceFromMask(
                ("[#f ]",),
            ),
            name="pinion_set",
        )
        mdb.models[model_name].parts["Pinion"].SectionAssignment(
            offset=0.0,
            offsetField="",
            offsetType=MIDDLE_SURFACE,
            region=mdb.models[model_name].parts["Pinion"].sets["pinion_set"],
            sectionName="Section-1",
            thicknessAssignment=FROM_SECTION,
        )

        # create Assembly
        mdb.models[model_name].rootAssembly.DatumCsysByDefault(CARTESIAN)
        mdb.models[model_name].rootAssembly.Instance(
            dependent=ON, name="Gear-1", part=mdb.models[model_name].parts["Gear"]
        )
        mdb.models[model_name].rootAssembly.Instance(
            dependent=ON, name="Pinion-1", part=mdb.models[model_name].parts["Pinion"]
        )

        # create load step
        mdb.models[model_name].StaticStep(
            adaptiveDampingRatio=0.05,
            continueDampingFactors=True,
            initialInc=0.01,
            maxInc=0.1,
            name="load",
            nlgeom=ON,
            previous="Initial",
            stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
        )

        # Mesh Gear
        mdb.models[model_name].parts["Gear"].setMeshControls(
            elemShape=QUAD,
            regions=mdb.models[model_name]
            .parts["Gear"]
            .faces.getSequenceFromMask(
                ("[#f ]",),
            ),
        )
        mdb.models[model_name].parts["Gear"].setElementType(
            elemTypes=(
                ElemType(elemCode=CPS8, elemLibrary=STANDARD),
                ElemType(elemCode=CPS6M, elemLibrary=STANDARD),
            ),
            regions=(
                mdb.models[model_name]
                .parts["Gear"]
                .faces.getSequenceFromMask(
                    ("[#f ]",),
                ),
            ),
        )
        mdb.models[model_name].parts["Gear"].seedEdgeBySize(
            constraint=FIXED,
            deviationFactor=0.1,
            edges=mdb.models[model_name]
            .parts["Gear"]
            .edges.getSequenceFromMask(
                ("[#100 ]",),
            ),
            size=0.05,
        )
        mdb.models[model_name].parts["Gear"].Set(
            edges=mdb.models[model_name]
            .parts["Gear"]
            .edges.getSequenceFromMask(
                ("[#100 ]",),
            ),
            name="gear_edge",
        )
        mdb.models[model_name].parts["Gear"].seedEdgeBySize(
            deviationFactor=0.1,
            edges=mdb.models[model_name]
            .parts["Gear"]
            .edges.getSequenceFromMask(
                ("[#211 ]",),
            ),
            size=0.1,
        )
        mdb.models[model_name].parts["Gear"].seedEdgeByBias(
            biasMethod=SINGLE,
            end1Edges=mdb.models[model_name]
            .parts["Gear"]
            .edges.getSequenceFromMask(
                ("[#80 ]",),
            ),
            maxSize=1.0,
            minSize=0.05,
        )
        mdb.models[model_name].parts["Gear"].seedEdgeByBias(
            biasMethod=SINGLE,
            end2Edges=mdb.models[model_name]
            .parts["Gear"]
            .edges.getSequenceFromMask(
                ("[#2 ]",),
            ),
            maxSize=1.0,
            minSize=0.05,
        )
        mdb.models[model_name].parts["Gear"].seedEdgeBySize(
            deviationFactor=0.1,
            edges=mdb.models[model_name]
            .parts["Gear"]
            .edges.getSequenceFromMask(
                ("[#28 ]",),
            ),
            size=1.0,
        )
        mdb.models[model_name].parts["Gear"].seedPart(
            deviationFactor=0.1, minSizeFactor=0.1, size=1.0
        )
        mdb.models[model_name].parts["Gear"].generateMesh()

        # Mesh Pinion
        mdb.models[model_name].parts["Pinion"].setMeshControls(
            elemShape=QUAD,
            regions=mdb.models[model_name]
            .parts["Pinion"]
            .faces.getSequenceFromMask(
                ("[#f ]",),
            ),
        )
        mdb.models[model_name].parts["Pinion"].setElementType(
            elemTypes=(
                ElemType(elemCode=CPS8, elemLibrary=STANDARD),
                ElemType(elemCode=CPS6M, elemLibrary=STANDARD),
            ),
            regions=(
                mdb.models[model_name]
                .parts["Pinion"]
                .faces.getSequenceFromMask(
                    ("[#f ]",),
                ),
            ),
        )
        mdb.models[model_name].parts["Pinion"].seedEdgeBySize(
            constraint=FIXED,
            deviationFactor=0.1,
            edges=mdb.models[model_name]
            .parts["Pinion"]
            .edges.getSequenceFromMask(
                ("[#40 ]",),
            ),
            size=0.05,
        )
        mdb.models[model_name].parts["Pinion"].Set(
            edges=mdb.models[model_name]
            .parts["Pinion"]
            .edges.getSequenceFromMask(
                ("[#40 ]",),
            ),
            name="pinion_edge",
        )
        mdb.models[model_name].parts["Pinion"].seedEdgeBySize(
            deviationFactor=0.1,
            edges=mdb.models[model_name]
            .parts["Pinion"]
            .edges.getSequenceFromMask(
                ("[#31 ]",),
            ),
            size=0.1,
        )
        mdb.models[model_name].parts["Pinion"].seedEdgeByBias(
            biasMethod=SINGLE,
            end2Edges=mdb.models[model_name]
            .parts["Pinion"]
            .edges.getSequenceFromMask(
                ("[#80 ]",),
            ),
            maxSize=1.0,
            minSize=0.05,
        )
        mdb.models[model_name].parts["Pinion"].seedEdgeByBias(
            biasMethod=SINGLE,
            end1Edges=mdb.models[model_name]
            .parts["Pinion"]
            .edges.getSequenceFromMask(
                ("[#8 ]",),
            ),
            maxSize=1.0,
            minSize=0.05,
        )
        mdb.models[model_name].parts["Pinion"].seedEdgeBySize(
            deviationFactor=0.1,
            edges=mdb.models[model_name]
            .parts["Pinion"]
            .edges.getSequenceFromMask(
                ("[#202 ]",),
            ),
            size=1.0,
        )
        mdb.models[model_name].parts["Pinion"].seedPart(
            deviationFactor=0.1, minSizeFactor=0.1, size=1.0
        )
        mdb.models[model_name].parts["Pinion"].generateMesh()

        # RP points and coupling
        mdb.models[model_name].parts["Pinion"].DatumPointByCoordinate(
            coords=(0.0, -pinion_r - 5, 0.0)
        )
        mdb.models[model_name].parts["Gear"].DatumPointByCoordinate(
            coords=(0.0, gear_r + 5, 0.0)
        )
        mdb.models[model_name].rootAssembly.regenerate()
        mdb.models[model_name].rootAssembly.ReferencePoint(
            point=mdb.models[model_name].rootAssembly.instances["Pinion-1"].datums[16]
        )
        mdb.models[model_name].rootAssembly.ReferencePoint(
            point=mdb.models[model_name].rootAssembly.instances["Gear-1"].datums[16]
        )
        mdb.models[model_name].rootAssembly.features.changeKey(
            fromName="RP-1", toName="Pinion_RP"
        )
        mdb.models[model_name].rootAssembly.features.changeKey(
            fromName="RP-2", toName="Gear_RP"
        )
        mdb.models[model_name].Coupling(
            controlPoint=Region(
                referencePoints=(
                    mdb.models[model_name].rootAssembly.referencePoints[6],
                )
            ),
            couplingType=KINEMATIC,
            influenceRadius=WHOLE_SURFACE,
            localCsys=None,
            name="Pinion_coupling",
            surface=Region(
                side1Edges=mdb.models[model_name]
                .rootAssembly.instances["Pinion-1"]
                .edges.getSequenceFromMask(
                    mask=("[#504 ]",),
                )
            ),
            u1=ON,
            u2=ON,
            ur3=ON,
        )
        mdb.models[model_name].Coupling(
            controlPoint=Region(
                referencePoints=(
                    mdb.models[model_name].rootAssembly.referencePoints[7],
                )
            ),
            couplingType=KINEMATIC,
            influenceRadius=WHOLE_SURFACE,
            localCsys=None,
            name="Gear_coupling",
            surface=Region(
                side1Edges=mdb.models[model_name]
                .rootAssembly.instances["Gear-1"]
                .edges.getSequenceFromMask(
                    mask=("[#444 ]",),
                )
            ),
            u1=OFF,
            u2=ON,
            ur3=OFF,
        )

        # Create Interaction Property
        mdb.models[model_name].ContactProperty("IntProp-1")
        mdb.models[model_name].interactionProperties["IntProp-1"].TangentialBehavior(
            formulation=FRICTIONLESS
        )
        mdb.models[model_name].interactionProperties["IntProp-1"].NormalBehavior(
            allowSeparation=ON,
            clearanceAtZeroContactPressure=0.0,
            constraintEnforcementMethod=AUGMENTED_LAGRANGE,
            contactStiffness=DEFAULT,
            contactStiffnessScaleFactor=1.0,
            pressureOverclosure=HARD,
        )

        # Create Surface-to-Surface contact
        mdb.models[model_name].rootAssembly.Surface(
            name="m_Surf_pinion",
            side1Edges=mdb.models[model_name]
            .rootAssembly.instances["Pinion-1"]
            .edges.getSequenceFromMask(
                ("[#40 ]",),
            ),
        )
        mdb.models[model_name].rootAssembly.Surface(
            name="s_Surf_gear",
            side1Edges=mdb.models[model_name]
            .rootAssembly.instances["Gear-1"]
            .edges.getSequenceFromMask(
                ("[#100 ]",),
            ),
        )
        mdb.models[model_name].SurfaceToSurfaceContactStd(
            adjustMethod=NONE,
            clearanceRegion=None,
            createStepName="Initial",
            datumAxis=None,
            initialClearance=OMIT,
            interactionProperty="IntProp-1",
            master=mdb.models[model_name].rootAssembly.surfaces["m_Surf_pinion"],
            name="cylinder_contact",
            slave=mdb.models[model_name].rootAssembly.surfaces["s_Surf_gear"],
            sliding=SMALL,
            thickness=ON,
        )

        # Create Boundary Conditions
        mdb.models[model_name].EncastreBC(
            createStepName="Initial",
            localCsys=None,
            name="Encastre_pinion",
            region=Region(
                referencePoints=(
                    mdb.models[model_name].rootAssembly.referencePoints[6],
                )
            ),
        )
        mdb.models[model_name].DisplacementBC(
            amplitude=UNSET,
            createStepName="Initial",
            distributionType=UNIFORM,
            fieldName="",
            localCsys=None,
            name="y_disp_pinion",
            region=Region(
                edges=mdb.models[model_name]
                .rootAssembly.instances["Pinion-1"]
                .edges.getSequenceFromMask(
                    mask=("[#504 ]",),
                )
            ),
            u1=UNSET,
            u2=SET,
            ur3=UNSET,
        )
        mdb.models[model_name].DisplacementBC(
            amplitude=UNSET,
            createStepName="Initial",
            distributionType=UNIFORM,
            fieldName="",
            localCsys=None,
            name="y_free_gear",
            region=Region(
                referencePoints=(
                    mdb.models[model_name].rootAssembly.referencePoints[7],
                )
            ),
            u1=SET,
            u2=UNSET,
            ur3=SET,
        )

        # Create Force
        if selected_load_type == "F":
            mdb.models[model_name].ConcentratedForce(
                cf2=-load,
                createStepName="load",
                distributionType=UNIFORM,
                field="",
                localCsys=None,
                name="Normal_force",
                region=Region(
                    referencePoints=(
                        mdb.models[model_name].rootAssembly.referencePoints[7],
                    )
                ),
            )
        # Create Dispacement
        elif selected_load_type == "D":
            mdb.models[model_name].DisplacementBC(
                name="Displacement",
                createStepName="load",
                region=Region(
                    referencePoints=(
                        mdb.models[model_name].rootAssembly.referencePoints[7],
                    )
                ),
                u1=UNSET,
                u2=-load,
                ur3=UNSET,
                amplitude=UNSET,
                fixed=OFF,
                distributionType=UNIFORM,
                fieldName="",
                localCsys=None,
            )

        # Set up Field output
        mdb.models[model_name].fieldOutputRequests["F-Output-1"].setValues(
            variables=(
                "S",
                "PE",
                "PEEQ",
                "PEMAG",
                "LE",
                "U",
                "RF",
                "CF",
                "CSTRESS",
                "CDSTRESS",
                "CDISP",
                "CFORCE",
                "CSTATUS",
            )
        )

        # Create Job
        mdb.models[model_name].rootAssembly.regenerate()
        mdb.Job(
            atTime=None,
            contactPrint=OFF,
            description="",
            echoPrint=OFF,
            explicitPrecision=SINGLE,
            getMemoryFromAnalysis=True,
            historyPrint=OFF,
            memory=90,
            memoryUnits=PERCENTAGE,
            model=model_name,
            modelPrint=OFF,
            multiprocessingMode=DEFAULT,
            name=odb_name,
            nodalOutputPrecision=FULL,
            numCpus=4,
            numDomains=4,
            numGPUs=0,
            queue=None,
            resultsFormat=ODB,
            scratch="",
            type=ANALYSIS,
            userSubroutine="",
            waitHours=0,
            waitMinutes=0,
        ).writeInput()

        # Save odb name
        model_dict["odb_name"].append(odb_name + ".odb")

        # Write model data to CSV
        with open(database_csv_path, "ab") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row = {key: model_dict[key][-1] for key in fieldnames}
            writer.writerow(row)

        print("  Model created successfully")

        # Auto-save CAE file every N models
        if AUTO_SAVE_INTERVAL > 0 and models_created % AUTO_SAVE_INTERVAL == 0:
            save_path = model_name_prefix + "_partial"
            mdb.saveAs(pathName=save_path)
            print(
                "  Auto-saved CAE file: "
                + save_path
                + ".cae (after "
                + str(models_created)
                + " models)"
            )

    # Final save of CAE file
    save_path = model_name_prefix
    mdb.saveAs(pathName=save_path)
    print("\nSaved CAE file: " + save_path + ".cae")
    print("Model database saved: " + database_csv_path)
    print("Analysis setup complete!")


if __name__ == "__main__":
    main()
