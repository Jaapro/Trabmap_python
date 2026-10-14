import os
import gc
import numpy as np
import pandas as pd
import tifffile as tiff

from scipy import ndimage as ndi
from skimage import morphology, filters
import SimpleITK as sitk


# ============================================================
# DEFAULT SETTINGS — edit these in the config yaml file, not here
# ============================================================

# these are default values for the script. 
# these values are overwritten by their lower case equivalents in the
# config.yaml file

RUNNER_CSV = "data/trabmap_runner.csv"

INPUT_ROOT = "input"

OBJECT_COLUMN = "object"
SEGMENTED_COLUMN = "segmented"
VOXEL_SIZE_COLUMN = "voxel_size"

INPUT_FOLDER_SUFFIX = "_TRABCORT"
RAW_FOLDER_SUFFIX = "_RAW"
AUTO_SEGMENT_RAW = True
RAW_SEGMENTATION_METHOD = "otsu"  # "otsu", "multiotsu", or "manual"
RAW_MANUAL_THRESHOLD = None
RAW_MULTIOTSU_CLASSES = 3
RAW_BONE_CLASS_MIN = 2
RAW_USE_MEDIAN_FILTER = True
RAW_MEDIAN_SIZE_VOXELS = (1, 3, 3)
RAW_KEEP_LARGEST_COMPONENT = True
RAW_MIN_OBJECT_SIZE = 1000
RAW_SAVE_THRESHOLD_DEBUG = True

PERIOSTEAL_THRESHOLD = 1
ENDOSTEAL_THRESHOLD = 1

IMAGE_EXTENSIONS = (".tif", ".tiff")

MIN_BONE_OBJECT_SIZE = 1000
USE_MEDIAN_FILTER = False
MEDIAN_SIZE_VOXELS = (1, 3, 3)

CROP_TO_BONE_BBOX = True
BBOX_PADDING_VOXELS = 10

# ------------------------------------------------------------
# Outer FILL
# ------------------------------------------------------------

CAP_FOR_3D_FILL = True
CAP_THICKNESS_SLICES = 5

SURFACE_HOLE_CLOSE_RADIUS_MM = 0.60 # default is 0.60

# ------------------------------------------------------------
# Internal compartment
# ------------------------------------------------------------

# keep only the largest connected void in the internal compartment before dilation and erosion steps. 
# this prevents small holes in cortex missed by cortex protection to be included in the internal filled area
KEEP_LARGEST_INTERNAL_VOID_COMPONENT = True

# close tiny cortical holes before subtracting from FILL
# default 0.05, change to 0.10 or 0.15 if cortex isnt properly filled. 
# set to lower value if trabecular bone structure is verz dense
# if set too high, this will fill in trabecular bone and ruin your
# segmentation
PROTECTED_BONE_CLOSE_RADIUS_MM = 0.05 

# Close marrow/intertrabecular spaces after subtraction.
MARROW_CLOSE_RADIUS_MM = 0.70

INTERNAL_FILL_MODE = "3d"  # "3d" or "2d"

KEEP_LARGEST_TRABFILL_COMPONENT = True

# Erosion and dilation steps can slightly change the geometry of trabfill,
# causing thin parts of the cortex to be assigned to trabecular mask. 
# this setting removes thin sheets of pixels sometimes accidentally 
# assigned to TRAB and assigns them to cort
REMOVE_THIN_TRAB_EDGE_LINES = True 
THIN_TRAB_OPEN_RADIUS_VOXELS = 2

SAVE_MIDDLE_SLICE_DEBUGS = True

# Folder/output naming. These are configurable so users never need to edit
# this script when changing the project folder layout or segmentation suffixes.
SEGMENTATIONS_DIR_NAME = "segmentations"
DEBUG_FOLDER_SUFFIX = "_DEBUG"
MANUAL_FILL_FOLDER_SUFFIX = "_FILL"

OUTPUT_FILL_SUFFIX = "_FILL"
OUTPUT_CORT_SUFFIX = "_CORT"
OUTPUT_TRAB_SUFFIX = "_TRAB"
OUTPUT_MARROW_SUFFIX = "_MARROW"
OUTPUT_TRABFILL_SUFFIX = "_TRABFILL"

OUTPUT_IMAGE_EXTENSION = ".tif"
OUTPUT_FOREGROUND_VALUE = 255

# Raw user-supplied config values are stored here by run_from_config() so they
# can be written back to the runner CSV for reproducibility.
CONFIG_VALUES = {}


def validate_settings():
    """Catch configuration mistakes before starting a long segmentation run."""
    if str(RAW_SEGMENTATION_METHOD).strip().lower() not in ["otsu", "multiotsu", "manual"]:
        raise ValueError("raw_segmentation_method must be otsu, multiotsu, or manual.")
    if str(INTERNAL_FILL_MODE).strip().lower() not in ["3d", "2d"]:
        raise ValueError("internal_fill_mode must be 3d or 2d.")
    if int(RAW_MULTIOTSU_CLASSES) < 2:
        raise ValueError("raw_multiotsu_classes must be >= 2.")
    if int(RAW_MIN_OBJECT_SIZE) < 0 or int(MIN_BONE_OBJECT_SIZE) < 0:
        raise ValueError("Object-size thresholds must be >= 0.")
    if int(BBOX_PADDING_VOXELS) < 0:
        raise ValueError("bbox_padding_voxels must be >= 0.")
    if int(CAP_THICKNESS_SLICES) < 1:
        raise ValueError("cap_thickness_slices must be >= 1.")
    if int(THIN_TRAB_OPEN_RADIUS_VOXELS) < 1:
        raise ValueError("thin_trab_open_radius_voxels must be >= 1.")
    if str(OUTPUT_IMAGE_EXTENSION).lower() not in [".tif", ".tiff"]:
        raise ValueError("output_image_extension must be .tif or .tiff.")
    if not 0 <= int(OUTPUT_FOREGROUND_VALUE) <= 255:
        raise ValueError("output_foreground_value must be between 0 and 255.")
    if not IMAGE_EXTENSIONS:
        raise ValueError("image_extensions must contain at least one extension.")




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
# STACK
# ============================================================

def read_stack(folder, image_extensions=None):
    # IMAGE_EXTENSIONS is a config setting. Do not duplicate a private default
    # here, otherwise changing image_extensions in YAML has no effect.
    if image_extensions is None:
        image_extensions = IMAGE_EXTENSIONS
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

    return np.stack(stack, axis=0)


def save_stack_from_crop(mask_crop, full_shape, bbox, folder, prefix):
    os.makedirs(folder, exist_ok=True)

    z_slice, y_slice, x_slice = bbox
    z0 = z_slice.start
    z1 = z_slice.stop

    for z in range(full_shape[0]):
        out = np.zeros((full_shape[1], full_shape[2]), dtype=np.uint8)

        if z0 <= z < z1:
            crop_z = z - z0
            out[y_slice, x_slice] = mask_crop[crop_z].astype(np.uint8) * int(OUTPUT_FOREGROUND_VALUE)

        filename = f"{prefix}_{z:04d}{OUTPUT_IMAGE_EXTENSION}"
        tiff.imwrite(os.path.join(folder, filename), out)


def save_middle_slices(mask, debug_folder, name):
    if not SAVE_MIDDLE_SLICE_DEBUGS or debug_folder is None:
        return

    os.makedirs(debug_folder, exist_ok=True)

    mask_uint8 = mask.astype(np.uint8) * int(OUTPUT_FOREGROUND_VALUE)

    z_mid = mask_uint8.shape[0] // 2
    y_mid = mask_uint8.shape[1] // 2
    x_mid = mask_uint8.shape[2] // 2

    tiff.imwrite(os.path.join(debug_folder, f"{name}_Zmid{OUTPUT_IMAGE_EXTENSION}"), mask_uint8[z_mid])
    tiff.imwrite(os.path.join(debug_folder, f"{name}_Ymid{OUTPUT_IMAGE_EXTENSION}"), mask_uint8[:, y_mid, :])
    tiff.imwrite(os.path.join(debug_folder, f"{name}_Xmid{OUTPUT_IMAGE_EXTENSION}"), mask_uint8[:, :, x_mid])


def save_binary_stack(mask, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    mask_u8 = mask.astype(np.uint8) * int(OUTPUT_FOREGROUND_VALUE)
    for z in range(mask_u8.shape[0]):
        tiff.imwrite(os.path.join(folder, f"{prefix}_{z:04d}{OUTPUT_IMAGE_EXTENSION}"), mask_u8[z])


def save_raw_threshold_debug(raw_volume, binary_mask, debug_folder, object_name):
    if not RAW_SAVE_THRESHOLD_DEBUG or debug_folder is None:
        return
    os.makedirs(debug_folder, exist_ok=True)

    def norm8(arr):
        arr = np.asarray(arr, dtype=np.float32)
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        return (np.clip((arr-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)

    raw_u8 = norm8(raw_volume)
    mask_u8 = binary_mask.astype(np.uint8) * int(OUTPUT_FOREGROUND_VALUE)
    zmid, ymid, xmid = [s // 2 for s in raw_volume.shape]
    for plane, rs, ms in [
        ("Zmid", raw_u8[zmid], mask_u8[zmid]),
        ("Ymid", raw_u8[:, ymid, :], mask_u8[:, ymid, :]),
        ("Xmid", raw_u8[:, :, xmid], mask_u8[:, :, xmid]),
    ]:
        tiff.imwrite(os.path.join(debug_folder, f"{object_name}_RAW_{plane}{OUTPUT_IMAGE_EXTENSION}"), rs)
        tiff.imwrite(os.path.join(debug_folder, f"{object_name}_RAW_segmented_{plane}{OUTPUT_IMAGE_EXTENSION}"), ms)


def segment_raw_volume(raw_volume):
    method = str(RAW_SEGMENTATION_METHOD).strip().lower()
    raw = np.asarray(raw_volume)
    vals = raw[np.isfinite(raw)]
    if vals.size == 0:
        raise RuntimeError("RAW volume contains no finite intensity values.")

    if method == "otsu":
        threshold = float(filters.threshold_otsu(vals))
        bone = raw >= threshold
        threshold_used = threshold
    elif method == "multiotsu":
        classes = int(RAW_MULTIOTSU_CLASSES)
        thresholds = filters.threshold_multiotsu(vals, classes=classes)
        labels = np.digitize(raw, bins=thresholds)
        min_class = int(RAW_BONE_CLASS_MIN)
        if not 0 <= min_class < classes:
            raise RuntimeError(f"RAW_BONE_CLASS_MIN must be between 0 and {classes-1}.")
        bone = labels >= min_class
        threshold_used = ",".join(str(float(x)) for x in thresholds)
    elif method == "manual":
        if RAW_MANUAL_THRESHOLD is None:
            raise RuntimeError("RAW_MANUAL_THRESHOLD must be set for manual thresholding.")
        threshold = float(RAW_MANUAL_THRESHOLD)
        bone = raw >= threshold
        threshold_used = threshold
    else:
        raise RuntimeError("RAW_SEGMENTATION_METHOD must be 'otsu', 'multiotsu', or 'manual'.")

    bone = bone.astype(bool)
    if RAW_USE_MEDIAN_FILTER:
        bone = ndi.median_filter(bone.astype(np.uint8), size=tuple(RAW_MEDIAN_SIZE_VOXELS)).astype(bool)
    if int(RAW_MIN_OBJECT_SIZE) > 0:
        bone = remove_small_objects_3d(bone, int(RAW_MIN_OBJECT_SIZE))
    if RAW_KEEP_LARGEST_COMPONENT:
        bone = largest_component(bone)
    if not np.any(bone):
        raise RuntimeError("RAW segmentation produced an empty mask.")
    return bone, threshold_used


# ============================================================
# FOLDER FINDERS
# ============================================================

def find_raw_folder(object_name):
    object_folder = os.path.join(INPUT_ROOT, object_name)
    if not os.path.exists(object_folder):
        print(f"  Object folder missing: {object_folder}")
        return None
    subfolders = [f for f in os.listdir(object_folder) if os.path.isdir(os.path.join(object_folder, f))]
    candidates = [f for f in subfolders if f.endswith(RAW_FOLDER_SUFFIX)]
    if not candidates:
        print(f"  No *{RAW_FOLDER_SUFFIX} folder found.")
        return None
    if len(candidates) > 1:
        print(f"  WARNING: multiple *{RAW_FOLDER_SUFFIX} folders found. Using: {candidates[0]}")
    return os.path.join(object_folder, candidates[0])


def ensure_trabcort_folder(object_name, debug_folder=None):
    existing = find_trabcort_folder(object_name)
    if existing is not None:
        print("  WARNING: existing TRABCORT folder found; automatic RAW segmentation skipped.")
        return existing, "existing", None
    if not AUTO_SEGMENT_RAW:
        print("  No TRABCORT folder found and AUTO_SEGMENT_RAW is disabled.")
        return None, "disabled", None
    raw_folder = find_raw_folder(object_name)
    if raw_folder is None:
        print("  No TRABCORT or RAW folder found.")
        return None, "missing", None

    print(f"  RAW folder found: {raw_folder}")
    raw_volume = read_stack(raw_folder)
    print(f"  RAW volume shape: {raw_volume.shape}; dtype: {raw_volume.dtype}")
    print(f"  RAW intensity range: {float(np.nanmin(raw_volume))} to {float(np.nanmax(raw_volume))}")
    trabcort_mask, threshold_used = segment_raw_volume(raw_volume)

    output_name = object_name + INPUT_FOLDER_SUFFIX
    output_folder = os.path.join(INPUT_ROOT, object_name, output_name)
    if os.path.exists(output_folder):
        print("  WARNING: TRABCORT folder appeared before save; using existing folder.")
        return output_folder, "existing", None

    print(f"  Saving automatically segmented TRABCORT stack: {output_folder}")
    save_binary_stack(trabcort_mask, output_folder, output_name)
    save_raw_threshold_debug(raw_volume, trabcort_mask, debug_folder, object_name)
    del raw_volume, trabcort_mask
    gc.collect()
    return output_folder, str(RAW_SEGMENTATION_METHOD).strip().lower(), threshold_used


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
        if f.endswith(MANUAL_FILL_FOLDER_SUFFIX)
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print(f"  WARNING: multiple *{MANUAL_FILL_FOLDER_SUFFIX} folders found.")
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
    # Step 2: protected bone small (holes closed in cortex)
    # silly name, should change
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
    # Step 4: generate mutually exclusive masks
    # --------------------------------------------------------

    print("  Creating final masks...")
    trabecular = real_bone & internal_filled
    cortical = real_bone & filled_bone & (~internal_filled)
    marrow_cavity = internal_filled & (~real_bone)

    if REMOVE_THIN_TRAB_EDGE_LINES:
        print("  Removing thin edge artifacts")
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
    """Log every user-supplied segmentation config value to the runner CSV."""
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
    validate_settings()
    print("\n=== Loading runner CSV ===")

    df = pd.read_csv(RUNNER_CSV, sep=None, engine="python")
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

        output_folder = os.path.join(INPUT_ROOT, object_name, SEGMENTATIONS_DIR_NAME)
        os.makedirs(output_folder, exist_ok=True)
        debug_folder = os.path.join(output_folder, object_name + DEBUG_FOLDER_SUFFIX)

        try:
            input_folder, raw_status, raw_threshold = ensure_trabcort_folder(
                object_name, debug_folder=debug_folder
            )
        except Exception as e:
            print("  RAW segmentation failed")
            print(e)
            continue

        if input_folder is None:
            continue

        if "raw_segmentation_status" not in df.columns:
            df["raw_segmentation_status"] = ""
        if "raw_segmentation_threshold" not in df.columns:
            df["raw_segmentation_threshold"] = ""
        df.loc[idx, "raw_segmentation_status"] = raw_status
        if raw_threshold is not None:
            df.loc[idx, "raw_segmentation_threshold"] = raw_threshold

        manual_fill_folder = find_optional_fill_folder(object_name)
        if manual_fill_folder is not None:
            print(f"  Manual FILL folder found: {manual_fill_folder}")
        else:
            print("  No manual FILL folder found. Will compute FILL automatically.")
        print(f"  Input folder: {input_folder}")

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
            "filled_bone": OUTPUT_FILL_SUFFIX,
            "cortical": OUTPUT_CORT_SUFFIX,
            "trabecular": OUTPUT_TRAB_SUFFIX,
            "marrow_cavity": OUTPUT_MARROW_SUFFIX,
            "internal_filled": OUTPUT_TRABFILL_SUFFIX,
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
            df.to_csv(RUNNER_CSV, index=False)
            print("  Updated runner CSV: segmented = yes")
        else:
            print("  WARNING: CSV appears to be open")
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
    """Run batch segmentation after applying a config dictionary."""
    global CONFIG_VALUES

    # Fail loudly on misspelled/obsolete YAML keys instead of silently using a
    # value embedded in this module.
    known_keys = {
        name.lower()
        for name, value in globals().items()
        if name.isupper() and name != "CONFIG_VALUES"
    }
    normalized_keys = {str(key).lower() for key in config}
    unknown_keys = sorted(normalized_keys - known_keys)
    if unknown_keys:
        raise ValueError(
            "Unknown segmentation config key(s): "
            + ", ".join(unknown_keys)
            + ". Update the YAML or add the setting explicitly to segmentation.py."
        )

    CONFIG_VALUES = dict(config)
    from .config import apply_config_to_module
    apply_config_to_module(__import__(__name__, fromlist=['*']), config)
    validate_settings()
    return main()
