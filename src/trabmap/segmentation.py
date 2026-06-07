# trabmap_batch_clean_simpleitk_segmentation.py

import os
import gc
import numpy as np
import pandas as pd
import tifffile as tiff

from scipy import ndimage as ndi
from skimage import morphology
import SimpleITK as sitk


# ============================================================
# USER SETTINGS
# ============================================================

RUNNER_CSV = "trabmap_runner.csv"

INPUT_ROOT = "input"

OBJECT_COLUMN = "object"
SEGMENTED_COLUMN = "segmented"
VOXEL_SIZE_COLUMN = "voxel_size"

INPUT_FOLDER_SUFFIX = "_TRABCORT"

PERIOSTEAL_THRESHOLD = 1
ENDOSTEAL_THRESHOLD = 1

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

MIN_BONE_OBJECT_SIZE = 1000
USE_MEDIAN_FILTER = True
MEDIAN_SIZE_VOXELS = (1, 3, 3)

CROP_TO_BONE_BBOX = True
BBOX_PADDING_VOXELS = 10

# ------------------------------------------------------------
# Outer FILL
# ------------------------------------------------------------

CAP_FOR_3D_FILL = True
CAP_THICKNESS_SLICES = 5

SURFACE_HOLE_CLOSE_RADIUS_MM = 0.70 # default is 0.70

# ------------------------------------------------------------
# Internal compartment
# ------------------------------------------------------------

# keep only the largest connected void in the internal compartment before dilation and erosion steps. prevents small holes in cortex missed by cortex protection to be included in the internal filled area
KEEP_LARGEST_INTERNAL_VOID_COMPONENT = True

# Lightly close tiny cortical holes before subtracting from FILL
# default 0.05, change to 0.10 or 0.15 if cortex isnt properly filled. 
# set to lower value if trabecular bone structure is verz dense
PROTECTED_BONE_CLOSE_RADIUS_MM = 0.05 

# Close marrow/intertrabecular spaces after subtraction.
MARROW_CLOSE_RADIUS_MM = 0.60

INTERNAL_FILL_MODE = "3d"  # "3d" or "2d"

KEEP_LARGEST_TRABFILL_COMPONENT = True

# Erosion and dilation steps can slightly change the geometry of trabfill, causing thin parts of the cortex to be assigned to trabecular mask. 
# this setting removes thin sheets of pixels sometimes accidentally assigned to TRAB
REMOVE_THIN_TRAB_EDGE_LINES = True 
THIN_TRAB_OPEN_RADIUS_VOXELS = 1

SAVE_MIDDLE_SLICE_DEBUGS = True

# Raw user-supplied config values are stored here by run_from_config() so they
# can be written back to the runner CSV for reproducibility.
CONFIG_VALUES = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def check_runner_csv_writable(csv_path):
    try:
        with open(csv_path, "a"):
            pass
        return True
    except PermissionError:
        return False


def get_voxel_size_mm(row):
    voxel_size = float(row[VOXEL_SIZE_COLUMN])

    if voxel_size <= 0:
        raise RuntimeError(f"Invalid voxel size: {voxel_size}")

    return voxel_size


def mm_to_voxel_radius(radius_mm, voxel_size_mm):
    return max(1, int(round(radius_mm / voxel_size_mm)))


def radius_vector_from_mm(radius_mm, voxel_size_mm):
    r = mm_to_voxel_radius(radius_mm, voxel_size_mm)
    return [r, r, r]


def largest_component(mask):
    labels, num = ndi.label(mask)

    if num == 0:
        return mask

    counts = np.bincount(labels.ravel())
    counts[0] = 0

    largest = counts.argmax()
    out = labels == largest

    del labels
    return out


def remove_small_objects_3d(mask, min_size):
    return morphology.remove_small_objects(
        mask.astype(bool),
        min_size=min_size
    )


def fill_internal_voids_2d(mask):
    filled = np.zeros_like(mask, dtype=bool)

    for z in range(mask.shape[0]):
        filled[z] = ndi.binary_fill_holes(mask[z])

    return filled


def fill_internal_voids(mask, mode="3d"):
    mode = mode.lower()

    if mode == "3d":
        return ndi.binary_fill_holes(mask)

    if mode == "2d":
        return fill_internal_voids_2d(mask)

    raise ValueError("INTERNAL_FILL_MODE must be '3d' or '2d'.")


# ============================================================
# SIMPLEITK HELPERS
# ============================================================

def numpy_bool_to_sitk(mask):
    return sitk.GetImageFromArray(mask.astype(np.uint8))


def sitk_to_numpy_bool(img):
    return sitk.GetArrayFromImage(img).astype(bool)


def sitk_binary_closing(mask, radius_mm, voxel_size_mm):
    radius = radius_vector_from_mm(radius_mm, voxel_size_mm)

    img = numpy_bool_to_sitk(mask)

    closed_img = sitk.BinaryMorphologicalClosing(
        img,
        radius,
        sitk.sitkBall,
        1.0,
        True
    )

    closed = sitk_to_numpy_bool(closed_img)

    del img, closed_img
    gc.collect()

    return closed


def sitk_binary_opening_voxels(mask, radius_voxels):
    img = numpy_bool_to_sitk(mask)

    opened_img = sitk.BinaryMorphologicalOpening(
        img,
        [radius_voxels, radius_voxels, radius_voxels],
        sitk.sitkBall,
        1.0
    )

    opened = sitk_to_numpy_bool(opened_img)

    del img, opened_img
    gc.collect()

    return opened


# ============================================================
# FILL HELPERS
# ============================================================

def cap_volume_ends(mask, cap_thickness=5):
    capped = mask.copy()

    nonempty = np.where(mask.reshape(mask.shape[0], -1).any(axis=1))[0]

    if len(nonempty) == 0:
        return capped

    first = int(nonempty[0])
    last = int(nonempty[-1])

    first_cap = ndi.binary_fill_holes(mask[first])
    last_cap = ndi.binary_fill_holes(mask[last])

    for z in range(first, min(first + cap_thickness, mask.shape[0])):
        capped[z] = capped[z] | first_cap

    for z in range(max(last - cap_thickness + 1, 0), last + 1):
        capped[z] = capped[z] | last_cap

    return capped


def fill_by_3d_exterior_connectivity(closed_shell):
    background = ~closed_shell
    labels, num = ndi.label(background)

    if num == 0:
        return closed_shell.copy()

    border_labels = set()

    border_labels.update(np.unique(labels[0, :, :]))
    border_labels.update(np.unique(labels[-1, :, :]))
    border_labels.update(np.unique(labels[:, 0, :]))
    border_labels.update(np.unique(labels[:, -1, :]))
    border_labels.update(np.unique(labels[:, :, 0]))
    border_labels.update(np.unique(labels[:, :, -1]))

    border_labels.discard(0)

    exterior = np.isin(labels, list(border_labels))

    filled = ~exterior

    del labels, exterior
    gc.collect()

    return filled


def make_filled_bone(bone_mask, voxel_size_mm):
    print("  Creating outer filled bone envelope...")

    shell = largest_component(bone_mask)

    if CAP_FOR_3D_FILL:
        print("  Capping stack ends...")
        shell = cap_volume_ends(shell, CAP_THICKNESS_SLICES)

    print("  Closing small surface holes...")

    shell = sitk_binary_closing(
        shell,
        SURFACE_HOLE_CLOSE_RADIUS_MM,
        voxel_size_mm
    )

    shell = largest_component(shell)

    print("  Filling enclosed 3D space...")

    filled_bone = fill_by_3d_exterior_connectivity(shell)
    filled_bone = largest_component(filled_bone)

    del shell
    gc.collect()

    return filled_bone


# ============================================================
# CROP HELPERS
# ============================================================

def crop_to_mask_bbox(volume, mask, padding=10):
    coords = np.argwhere(mask)

    if coords.size == 0:
        full_bbox = tuple(slice(0, s) for s in volume.shape)
        return volume, full_bbox

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1

    zmin = max(zmin - padding, 0)
    ymin = max(ymin - padding, 0)
    xmin = max(xmin - padding, 0)

    zmax = min(zmax + padding, volume.shape[0])
    ymax = min(ymax + padding, volume.shape[1])
    xmax = min(xmax + padding, volume.shape[2])

    bbox = (
        slice(zmin, zmax),
        slice(ymin, ymax),
        slice(xmin, xmax)
    )

    return volume[bbox], bbox


# ============================================================
# STACK I/O
# ============================================================

def read_stack(folder, image_extensions=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
    image_extensions = tuple(image_extensions)

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(image_extensions)
    ])

    if len(files) == 0:
        raise RuntimeError(f"No image files found in: {folder}")

    stack = []

    for f in files:
        img = tiff.imread(os.path.join(folder, f))
        stack.append(img)

    return np.stack(stack, axis=0).astype(np.uint8)


def save_stack_from_crop(mask_crop, full_shape, bbox, folder, prefix):
    os.makedirs(folder, exist_ok=True)

    z_slice, y_slice, x_slice = bbox
    z0 = z_slice.start
    z1 = z_slice.stop

    for z in range(full_shape[0]):
        out = np.zeros((full_shape[1], full_shape[2]), dtype=np.uint8)

        if z0 <= z < z1:
            crop_z = z - z0
            out[y_slice, x_slice] = mask_crop[crop_z].astype(np.uint8) * 255

        filename = f"{prefix}_{z:04d}.tif"
        tiff.imwrite(os.path.join(folder, filename), out)


def save_middle_slices(mask, debug_folder, name):
    if not SAVE_MIDDLE_SLICE_DEBUGS or debug_folder is None:
        return

    os.makedirs(debug_folder, exist_ok=True)

    mask_uint8 = mask.astype(np.uint8) * 255

    z_mid = mask_uint8.shape[0] // 2
    y_mid = mask_uint8.shape[1] // 2
    x_mid = mask_uint8.shape[2] // 2

    tiff.imwrite(os.path.join(debug_folder, f"{name}_Zmid.tif"), mask_uint8[z_mid])
    tiff.imwrite(os.path.join(debug_folder, f"{name}_Ymid.tif"), mask_uint8[:, y_mid, :])
    tiff.imwrite(os.path.join(debug_folder, f"{name}_Xmid.tif"), mask_uint8[:, :, x_mid])


# ============================================================
# FOLDER FINDERS
# ============================================================

def find_trabcort_folder(object_name):
    object_folder = os.path.join(INPUT_ROOT, object_name)

    if not os.path.exists(object_folder):
        print(f"  Object folder missing: {object_folder}")
        return None

    subfolders = [
        f for f in os.listdir(object_folder)
        if os.path.isdir(os.path.join(object_folder, f))
    ]

    candidates = [
        f for f in subfolders
        if f.endswith(INPUT_FOLDER_SUFFIX)
    ]

    if len(candidates) == 0:
        print(f"  No *{INPUT_FOLDER_SUFFIX} folder found.")
        return None

    if len(candidates) > 1:
        print(f"  WARNING: multiple *{INPUT_FOLDER_SUFFIX} folders found.")
        print(f"  Using: {candidates[0]}")

    return os.path.join(object_folder, candidates[0])


def find_optional_fill_folder(object_name):
    object_folder = os.path.join(INPUT_ROOT, object_name)

    if not os.path.exists(object_folder):
        return None

    subfolders = [
        f for f in os.listdir(object_folder)
        if os.path.isdir(os.path.join(object_folder, f))
    ]

    candidates = [
        f for f in subfolders
        if f.endswith("_FILL")
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print("  WARNING: multiple *_FILL folders found.")
        print(f"  Using: {candidates[0]}")

    return os.path.join(object_folder, candidates[0])


# ============================================================
# SEGMENTATION
# ============================================================

def segment_bone(
    volume,
    voxel_size_mm,
    debug_folder=None,
    manual_fill_folder=None
):
    full_shape = volume.shape

    print("  Thresholding binary bone stack...")

    original_bone = volume >= PERIOSTEAL_THRESHOLD

    if CROP_TO_BONE_BBOX:
        print("  Cropping to bone bounding box...")

        volume, bbox = crop_to_mask_bbox(
            volume,
            original_bone,
            padding=BBOX_PADDING_VOXELS
        )

        print(f"  Cropped shape: {volume.shape}")

        original_bone = volume >= PERIOSTEAL_THRESHOLD

    else:
        bbox = tuple(slice(0, s) for s in volume.shape)

    original_bone = remove_small_objects_3d(
        original_bone,
        min_size=MIN_BONE_OBJECT_SIZE
    )

    original_bone = largest_component(original_bone)

    if USE_MEDIAN_FILTER:

        print("  Median filtering original bone mask...")

        original_bone = ndi.median_filter(
            original_bone.astype(np.uint8),
            size=MEDIAN_SIZE_VOXELS
        ).astype(bool)

    real_bone = volume >= ENDOSTEAL_THRESHOLD

    # --------------------------------------------------------
    # Step 1: outer FILL
    # --------------------------------------------------------

    if manual_fill_folder is not None:
        print("  Using manual FILL stack...")

        manual_fill = read_stack(manual_fill_folder) > 0

        if manual_fill.shape != full_shape:
            raise RuntimeError(
                f"Manual FILL shape {manual_fill.shape} does not match input shape {full_shape}"
            )

        filled_bone = manual_fill[bbox].astype(bool) if CROP_TO_BONE_BBOX else manual_fill.astype(bool)

        del manual_fill

    else:
        filled_bone = make_filled_bone(
            original_bone,
            voxel_size_mm
        )

    # Always include real bone in FILL
    filled_bone = filled_bone | real_bone

    gc.collect()

    # --------------------------------------------------------
    # Step 2: protected bone
    # --------------------------------------------------------

    print("  Closing small holes ...")

    protected_bone = sitk_binary_closing(
        real_bone,
        PROTECTED_BONE_CLOSE_RADIUS_MM,
        voxel_size_mm
    )

    protected_bone = protected_bone & filled_bone

    # --------------------------------------------------------
    # Step 3: internal TRABFILL
    # --------------------------------------------------------

    print("  Estimating internal void space as FILL minus protected bone...")

    internal_void = filled_bone & (~protected_bone)

    if KEEP_LARGEST_INTERNAL_VOID_COMPONENT:
        print("  Keeping largest internal void component before marrow closing...")

        internal_void = largest_component(internal_void)
        internal_void = internal_void & filled_bone

    print("  Closing internal void space")

    internal_filled = sitk_binary_closing(
        internal_void,
        MARROW_CLOSE_RADIUS_MM,
        voxel_size_mm
    )

    print(f"  Filling internal compartment using {INTERNAL_FILL_MODE} fill...")

    internal_filled = fill_internal_voids(
        internal_filled,
        mode=INTERNAL_FILL_MODE
    )

    internal_filled = internal_filled & filled_bone

    if KEEP_LARGEST_TRABFILL_COMPONENT:
        print("  Keeping largest TRABFILL component...")

        internal_filled = largest_component(internal_filled)
        internal_filled = internal_filled & filled_bone

    # --------------------------------------------------------
    # Step 4: final mutually exclusive masks
    # --------------------------------------------------------

    print("  Creating final mutually exclusive compartment masks...")

    trabecular = real_bone & internal_filled

    cortical = real_bone & filled_bone & (~internal_filled)

    marrow_cavity = internal_filled & (~real_bone)

    if REMOVE_THIN_TRAB_EDGE_LINES:
        print("  Removing edge artifacts...")

        trabecular_clean = sitk_binary_opening_voxels(
            trabecular,
            THIN_TRAB_OPEN_RADIUS_VOXELS
        )

        trabecular_clean = trabecular_clean & real_bone & internal_filled

        removed_from_trab = trabecular & (~trabecular_clean)

        trabecular = trabecular_clean

        cortical = cortical | removed_from_trab

        cortical = cortical & (~trabecular)

        del trabecular_clean, removed_from_trab

    # Safety check: remove any possible overlap
    cortical = cortical & (~trabecular)

    save_middle_slices(filled_bone, debug_folder, "DEBUG_filled_bone")
    save_middle_slices(internal_filled, debug_folder, "DEBUG_internal_filled")
    save_middle_slices(cortical, debug_folder, "DEBUG_cortical")
    save_middle_slices(trabecular, debug_folder, "DEBUG_trabecular")
    save_middle_slices(marrow_cavity, debug_folder, "DEBUG_marrow_cavity")
    save_middle_slices(real_bone, debug_folder, "DEBUG_trabcort")
    save_middle_slices(protected_bone, debug_folder, "DEBUG_protected")
    save_middle_slices(internal_void, debug_folder, "DEBUG_internal_void")
    save_middle_slices(original_bone, debug_folder, "DEBUG_original_bone")

    del original_bone, real_bone, protected_bone, internal_void
    gc.collect()

    return {
        "filled_bone": filled_bone,
        "cortical": cortical,
        "trabecular": trabecular,
        "marrow_cavity": marrow_cavity,
        "internal_filled": internal_filled,
        "full_shape": full_shape,
        "bbox": bbox
    }



# ============================================================
# CONFIG LOGGING HELPERS
# ============================================================

def simple_config_value(value):
    """Convert config values to runner-CSV friendly text."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return value


def flatten_config_values(config, prefix=""):
    """Flatten a config dictionary/list into key-value pairs for runner CSV logging."""
    if not config:
        return {}
    flat = {}
    if isinstance(config, dict):
        for key, value in config.items():
            safe_key = str(key).strip().replace(" ", "_").replace("-", "_")
            new_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
            flat.update(flatten_config_values(value, new_prefix))
    elif isinstance(config, (list, tuple)):
        flat[prefix] = simple_config_value(config)
    else:
        flat[prefix] = simple_config_value(config)
    return flat


def write_config_values_to_runner_row(df, idx):
    """Write every user-supplied segmentation config value to the runner CSV."""
    if not CONFIG_VALUES:
        return
    for key, value in flatten_config_values(CONFIG_VALUES).items():
        col = f"config_segmentation_{key}"
        if col not in df.columns:
            df[col] = ""
        df.loc[idx, col] = simple_config_value(value)


# ============================================================
# MAIN BATCH LOOP
# ============================================================

def main():
    print("\n=== Loading runner CSV ===")

    df = pd.read_csv(RUNNER_CSV, sep=";")
    df.columns = df.columns.str.strip()

    print("CSV columns found:")
    print(df.columns.tolist())

    if SEGMENTED_COLUMN not in df.columns:
        df[SEGMENTED_COLUMN] = ""

    for idx, row in df.iterrows():
        object_name = str(row[OBJECT_COLUMN]).strip()

        segmented_status = str(row.get(SEGMENTED_COLUMN, "")).strip().lower()

        print("\n================================================")
        print(f"OBJECT: {object_name}")

        if segmented_status == "yes":
            print("  Already segmented. Skipping.")
            continue

        try:
            voxel_size_mm = get_voxel_size_mm(row)
        except Exception as e:
            print("  Could not read voxel size. Skipping.")
            print(f"  {e}")
            continue

        input_folder = find_trabcort_folder(object_name)

        if input_folder is None:
            continue

        manual_fill_folder = find_optional_fill_folder(object_name)

        if manual_fill_folder is not None:
            print(f"  Manual FILL folder found: {manual_fill_folder}")
        else:
            print("  No manual FILL folder found. Will compute FILL automatically.")

        print(f"  Input folder: {input_folder}")

        output_folder = os.path.join(
            INPUT_ROOT,
            object_name,
            "segmentations"
        )
        
        os.makedirs(output_folder, exist_ok=True)

        debug_folder = os.path.join(output_folder, object_name + "_DEBUG")

        try:
            print("  Reading stack...")
            volume = read_stack(input_folder)
            print(f"  Volume shape: {volume.shape}")
            print(f"  Volume dtype: {volume.dtype}")

        except Exception as e:
            print("  FAILED TO LOAD STACK")
            print(e)
            continue

        try:
            results = segment_bone(
                volume,
                voxel_size_mm,
                debug_folder=debug_folder,
                manual_fill_folder=manual_fill_folder
            )

        except Exception as e:
            print("  SEGMENTATION FAILED")
            print(e)
            continue

        print("  Saving outputs...")

        full_shape = results["full_shape"]
        bbox = results["bbox"]

        output_name_map = {
            "filled_bone": "_FILL",
            "cortical": "_CORT",
            "trabecular": "_TRAB",
            "marrow_cavity": "_MARROW",
            "internal_filled": "_TRABFILL"
        }

        for mask_name, suffix in output_name_map.items():
            output_stack_name = object_name + suffix
            outdir = os.path.join(output_folder, output_stack_name)

            save_stack_from_crop(
                results[mask_name],
                full_shape,
                bbox,
                outdir,
                output_stack_name
            )

        df.loc[idx, SEGMENTED_COLUMN] = "yes"
        write_config_values_to_runner_row(df, idx)

        if check_runner_csv_writable(RUNNER_CSV):
            df.to_csv(RUNNER_CSV, index=False, sep=";")
            print("  Updated runner CSV: segmented = yes")
        else:
            print("  WARNING: CSV appears to be open in Excel.")
            print("  Could not update segmented column.")

        del results, volume
        gc.collect()

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()


# ============================================================
# CONFIG-FRIENDLY ENTRY POINT
# ============================================================

def run_from_config(config):
    """Run batch segmentation after applying a config dictionary.

    This keeps the original tested functions intact, but removes the need to edit
    paths or parameters inside the source code. Config keys may be lower-case
    YAML names such as `runner_csv` or the original uppercase names such as
    `RUNNER_CSV`.
    """
    global CONFIG_VALUES
    CONFIG_VALUES = dict(config)
    from .config import apply_config_to_module
    apply_config_to_module(__import__(__name__, fromlist=['*']), config)
    return main()
