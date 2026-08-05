"""
Compare Medtool and Trabmap segmentation stacks.

For each phase (CORT, TRAB, TRABFILL, FILL), this script:
1. Finds and reads the corresponding image stack from a Medtool root folder
   and a Trabmap root folder.
2. Binarizes both stacks.
3. Calculates slice-by-slice overlap and difference metrics.
4. Writes:
      - segmentation_comparison_by_slice.csv
      - segmentation_comparison_summary.csv
5. Saves a categorical RGB difference stack for each phase:
      black = methods agree
      red   = Medtool foreground only
      blue  = Trabmap foreground only
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi


# =============================================================================
# USER SETTINGS
# =============================================================================

MEDTOOL_ROOT = Path(r"C:\Users\Jaaps\Documents\Trabmap\trabmap_github\input\TxState_D01_2011_Tibia_Dist_L\Output_using_medtool_seg\segmentations")
TRABMAP_ROOT = Path(r"C:\Users\Jaaps\Documents\Trabmap\trabmap_github\input\TxState_D01_2011_Tibia_Dist_L\segmentations")
OUTPUT_ROOT = Path(r"C:\Users\Jaaps\Documents\Trabmap\trabmap_github\input\TxState_D01_2011_Tibia_Dist_L\trabmap_medtool_segmentation_comparisons")

PHASE_SUFFIXES = {
    "CORT": "_CORT",
    "TRAB": "_TRAB",
    "TRABFILL": "_TRABFILL",
    "FILL": "_FILL",
}

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
BINARY_THRESHOLD = 0
REMOVE_SMALL_OBJECTS = False
MIN_OBJECT_SIZE_VOXELS = 10
REQUIRE_IDENTICAL_SHAPE = True
REQUIRE_IDENTICAL_SLICE_COUNT = True
SAVE_BINARY_XOR_STACK = True

COLOR_AGREEMENT = np.array([0, 0, 0], dtype=np.uint8)
COLOR_MEDTOOL_ONLY = np.array([255, 0, 0], dtype=np.uint8)
COLOR_TRABMAP_ONLY = np.array([0, 102, 255], dtype=np.uint8)


# =============================================================================
# FILE AND STACK HELPERS
# =============================================================================

def natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def image_files(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda p: natural_sort_key(p.name))
    if not files:
        raise RuntimeError(f"No supported image files found in: {folder}")
    return files


def find_phase_folder(root: Path, suffix: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    suffix_upper = suffix.upper()
    plain_name = suffix_upper.lstrip("_")

    def matches(path: Path) -> bool:
        name = path.name.upper()
        return name == plain_name or name == suffix_upper or name.endswith(suffix_upper)

    if root.is_dir() and matches(root):
        return root

    direct = [p for p in root.iterdir() if p.is_dir() and matches(p)]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise RuntimeError(
            f"Multiple folders matching {suffix} were found under {root}:\n" +
            "\n".join(f"  - {p}" for p in direct)
        )

    recursive = [p for p in root.rglob("*") if p.is_dir() and matches(p)]
    if len(recursive) == 1:
        return recursive[0]
    if not recursive:
        raise FileNotFoundError(f"No folder ending in {suffix} was found under: {root}")
    raise RuntimeError(
        f"Multiple folders matching {suffix} were found under {root}:\n" +
        "\n".join(f"  - {p}" for p in recursive)
    )


def read_binary_stack(folder: Path) -> tuple[np.ndarray, list[Path]]:
    files = image_files(folder)
    slices = []
    expected_shape = None

    for path in files:
        image = np.asarray(tiff.imread(path))
        if image.ndim == 3:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image in {path}, got shape {image.shape}")
        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(
                f"Inconsistent slice shape in {folder}: {path.name} has {image.shape}, "
                f"expected {expected_shape}"
            )
        slices.append(image > BINARY_THRESHOLD)

    stack = np.stack(slices, axis=0).astype(bool)

    if REMOVE_SMALL_OBJECTS:
        labels, number = ndi.label(stack)
        if number > 0:
            counts = np.bincount(labels.ravel())
            keep = counts >= MIN_OBJECT_SIZE_VOXELS
            keep[0] = False
            stack = keep[labels]

    return stack, files


def align_stacks(medtool: np.ndarray, trabmap: np.ndarray, phase: str):
    if medtool.shape == trabmap.shape:
        return medtool, trabmap
    if REQUIRE_IDENTICAL_SHAPE:
        raise ValueError(
            f"{phase}: stack shapes differ: Medtool={medtool.shape}, TrabMap={trabmap.shape}"
        )
    common_shape = tuple(min(a, b) for a, b in zip(medtool.shape, trabmap.shape))
    print(f"  WARNING: {phase} shapes differ. Cropping both to {common_shape}.")
    crop = tuple(slice(0, size) for size in common_shape)
    return medtool[crop], trabmap[crop]


# =============================================================================
# METRICS
# =============================================================================

def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else np.nan


def comparison_metrics(medtool: np.ndarray, trabmap: np.ndarray) -> dict[str, float | int]:
    medtool = medtool.astype(bool)
    trabmap = trabmap.astype(bool)

    both_fg = medtool & trabmap
    medtool_only = medtool & ~trabmap
    trabmap_only = trabmap & ~medtool
    both_bg = ~medtool & ~trabmap

    tp = int(np.count_nonzero(both_fg))
    fn = int(np.count_nonzero(medtool_only))
    fp = int(np.count_nonzero(trabmap_only))
    tn = int(np.count_nonzero(both_bg))

    medtool_count = tp + fn
    trabmap_count = tp + fp
    union_count = tp + fp + fn
    total = tp + fp + fn + tn
    different = fp + fn

    if medtool_count == 0 and trabmap_count == 0:
        dice = 1.0
        jaccard = 1.0
    else:
        dice = safe_divide(2 * tp, 2 * tp + fp + fn)
        jaccard = safe_divide(tp, union_count)

    return {
        "medtool_foreground_pixels": medtool_count,
        "trabmap_foreground_pixels": trabmap_count,
        "intersection_pixels": tp,
        "union_pixels": union_count,
        "medtool_only_pixels": fn,
        "trabmap_only_pixels": fp,
        "different_pixels": different,
        "same_pixels": tp + tn,
        "total_pixels": total,
        "dice": dice,
        "jaccard": jaccard,
        "pixel_accuracy": safe_divide(tp + tn, total),
        "sensitivity_recall_vs_medtool": safe_divide(tp, tp + fn),
        "specificity_vs_medtool": safe_divide(tn, tn + fp),
        "precision_vs_medtool": safe_divide(tp, tp + fp),
        "false_negative_rate_vs_medtool": safe_divide(fn, tp + fn),
        "false_positive_rate_vs_medtool": safe_divide(fp, fp + tn),
        "relative_foreground_difference": safe_divide(
            trabmap_count - medtool_count, medtool_count
        ),
        "absolute_foreground_difference_pixels": abs(trabmap_count - medtool_count),
        "percent_pixels_different": 100.0 * safe_divide(different, total),
    }


# =============================================================================
# DIFFERENCE IMAGE OUTPUT
# =============================================================================

def categorical_difference_image(medtool_slice: np.ndarray, trabmap_slice: np.ndarray) -> np.ndarray:
    medtool_only = medtool_slice & ~trabmap_slice
    trabmap_only = trabmap_slice & ~medtool_slice
    rgb = np.zeros((*medtool_slice.shape, 3), dtype=np.uint8)
    rgb[:] = COLOR_AGREEMENT
    rgb[medtool_only] = COLOR_MEDTOOL_ONLY
    rgb[trabmap_only] = COLOR_TRABMAP_ONLY
    return rgb


def save_difference_stacks(
    medtool: np.ndarray,
    trabmap: np.ndarray,
    phase: str,
    output_root: Path,
) -> None:
    rgb_folder = output_root / f"{phase}_DIFFERENCE_RGB"
    rgb_folder.mkdir(parents=True, exist_ok=True)

    xor_folder = output_root / f"{phase}_DIFFERENCE_XOR"
    if SAVE_BINARY_XOR_STACK:
        xor_folder.mkdir(parents=True, exist_ok=True)

    for z in range(medtool.shape[0]):
        rgb = categorical_difference_image(medtool[z], trabmap[z])
        tiff.imwrite(
            rgb_folder / f"{phase}_difference_{z:04d}.tif",
            rgb,
            photometric="rgb",
        )

        if SAVE_BINARY_XOR_STACK:
            xor_image = np.logical_xor(medtool[z], trabmap[z]).astype(np.uint8) * 255
            tiff.imwrite(xor_folder / f"{phase}_xor_{z:04d}.tif", xor_image)


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def compare_phase(phase: str, suffix: str):
    print(f"\n=== {phase} ===")

    medtool_folder = find_phase_folder(MEDTOOL_ROOT, suffix)
    trabmap_folder = find_phase_folder(TRABMAP_ROOT, suffix)

    print(f"  Medtool: {medtool_folder}")
    print(f"  TrabMap: {trabmap_folder}")

    medtool_stack, medtool_files = read_binary_stack(medtool_folder)
    trabmap_stack, trabmap_files = read_binary_stack(trabmap_folder)

    if REQUIRE_IDENTICAL_SLICE_COUNT and len(medtool_files) != len(trabmap_files):
        raise ValueError(
            f"{phase}: slice counts differ: Medtool={len(medtool_files)}, "
            f"TrabMap={len(trabmap_files)}"
        )

    medtool_stack, trabmap_stack = align_stacks(medtool_stack, trabmap_stack, phase)

    slice_rows = []
    for z in range(medtool_stack.shape[0]):
        row = {
            "phase": phase,
            "slice_index": z,
            "medtool_filename": medtool_files[z].name if z < len(medtool_files) else "",
            "trabmap_filename": trabmap_files[z].name if z < len(trabmap_files) else "",
        }
        row.update(comparison_metrics(medtool_stack[z], trabmap_stack[z]))
        slice_rows.append(row)

    whole_stack_metrics = comparison_metrics(medtool_stack, trabmap_stack)
    summary_row = {
        "phase": phase,
        "medtool_folder": str(medtool_folder),
        "trabmap_folder": str(trabmap_folder),
        "stack_shape_z_y_x": "x".join(str(v) for v in medtool_stack.shape),
    }
    summary_row.update(whole_stack_metrics)

    save_difference_stacks(
        medtool_stack,
        trabmap_stack,
        phase,
        OUTPUT_ROOT / "difference_stacks",
    )

    print(
        f"  Dice={whole_stack_metrics['dice']:.6f}, "
        f"Jaccard={whole_stack_metrics['jaccard']:.6f}, "
        f"different={whole_stack_metrics['percent_pixels_different']:.4f}%"
    )

    return slice_rows, summary_row


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_slice_rows = []
    summary_rows = []

    for phase, suffix in PHASE_SUFFIXES.items():
        phase_slice_rows, phase_summary = compare_phase(phase, suffix)
        all_slice_rows.extend(phase_slice_rows)
        summary_rows.append(phase_summary)

    slice_df = pd.DataFrame(all_slice_rows)
    summary_df = pd.DataFrame(summary_rows)

    slice_csv = OUTPUT_ROOT / "segmentation_comparison_by_slice.csv"
    summary_csv = OUTPUT_ROOT / "segmentation_comparison_summary.csv"

    slice_df.to_csv(slice_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== COMPLETE ===")
    print(f"Slice metrics: {slice_csv}")
    print(f"Whole-stack summary: {summary_csv}")
    print(f"Difference stacks: {OUTPUT_ROOT / 'difference_stacks'}")
    print("\nDifference image colors:")
    print("  black = methods agree")
    print("  red   = Medtool foreground only")
    print("  blue  = TrabMap foreground only")


if __name__ == "__main__":
    main()