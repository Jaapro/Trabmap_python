"""
Trabmap local script: calculates BV/TV, EDT derived trabecular thickness and spacing, 
structure tensor direction, ParaView VTP export, and optional tetrahedral VTK mesh export and surface meshes

The script reads trabmap_runner.csv, finds rows where the finished column value
is not equal to "yes", auto-detects the matching _TRAB and _TRABFILL folders, 
runs Trabmap, prefixes outputs with the object name, and writes the settings/status back to the runner CSV.

Notes:
- Input and filled-mask can be a folder of .tif/.tiff slices, a 3D .tif/.tiff stack, or a .npy file.
- Coordinates are assumed to be z,y,x in image arrays, and x,y,z in Paraview outputs.
- Voxel spacing is assumed to be in mm
"""

from __future__ import annotations

from types import SimpleNamespace
import csv
import gc
import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import structure_tensor

# Optional imports. These are only needed for selected outputs.
try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import pyvista as pv
except Exception:
    pv = None


# =============================================================================
# SETTINGS: EDIT THESE IN THE CONFIG FILE
# =============================================================================

SETTINGS = SimpleNamespace(
    # -------------------------------------------------------------------------
    # Batch runner paths
    # -------------------------------------------------------------------------
    # CSV file needs to have at least these columns: finished, object, voxel_size.
    # The uploaded example uses semicolons, which this script detects automatically.
    runner_csv="trabmap_runner.csv",

    # Folder containing a subfolder per object.
    input_root="input",

    # Output parent folder. Each object gets its own subfolder here. old code
    output_heatmap_root=None,

    # -------------------------------------------------------------------------
    # Optional anatomical landmarks
    # -------------------------------------------------------------------------
    # no longer used, will be  removed.
    # instead use the runner file to give two sets of x,y,z coordinates of long 
    # axis and one set for the rotational axis
    landmarks_csv=None,

    # -------------------------------------------------------------------------
    # Optional custom anatomical axis settings
    # -------------------------------------------------------------------------
    # These are read from trabmap_runner.csv columns:
    # landmark_long_1_x, landmark_long_1_y, landmark_long_1_z,
    # landmark_long_2_x, landmark_long_2_y, landmark_long_2_z,
    # landmark_rotate_x, landmark_rotate_y, landmark_rotate_z.
    #
    # The two long-axis landmark coordinate sets define anatomical long axis.
    # this can be used to determine the orientation of the trabecular structure
    # relative to a specified long axis defined by two points and a rotation axis
    # defined byy another point
    # example: long axis can define the superior/inferior axis while the 
    # rotation axis can be used to specify the anterior/posteior axis
    # The rotate landmark defines the orientation of axes 2 and 3 around that
    # long axis. If all nine values are present, primary_dir_st_anat_1/2/3 are
    # calculated in this custom frame, and angle_to_custom_long_axis_deg is also
    # calculated. If the columns are absent or blank, no custom anatomical frame
    # is used for that object.
    use_custom_anatomical_frame=False,
    custom_long_axis_p1_xyz=None,
    custom_long_axis_p2_xyz=None,
    custom_rotation_point_xyz=None,

    # -------------------------------------------------------------------------
    # Sphere/grid settings
    # -------------------------------------------------------------------------
    # Spacing is read from the voxel_size column of the runner CSV and copied to
    # spacing_z, spacing_y, and spacing_x for each object.
    sphere_radius=2.5, # default is 2.5
    grid_step=2.5, # default is 2.5
    margin=None,

    # What to do with grid centers outside the filled mask:
    # "skip_outside" = ignore them; "zero_outside" = keep them with zero values.
    # skip outside is a sensible default, zero outside is to make output
    # comparable to ORS Dragonfly (but is not advised)
    center_inclusion_mode="skip_outside",

    # -------------------------------------------------------------------------
    # Boundary handling for non-BV/TV measurements
    # -------------------------------------------------------------------------
    # BV/TV already uses only TRABFILL voxels in the denominator. These settings
    # apply similar TRABFILL sphere clipping to trabecular thickness and structure-tensor
    # anisotropy so that voxels outside the internal TRABFILL compartment do
    # not contribute to these local measurements.
    clip_non_bvtv_to_filled_mask=True,
    ignore_trabfill_boundary_for_thickness=True,
    save_boundary_diagnostics=True,

    # -------------------------------------------------------------------------
    # Trabecular spacing settings
    # -------------------------------------------------------------------------
    # EDTSpacing is calculated as 2 * EDT(marrow space inside TRABFILL), analogous
    # to the local trabecular thickness calculation. When
    # ignore_trabfill_boundary_for_spacing=True, voxels outside TRABFILL do not
    # act as biologically meaningful background, reducing boundary bias near
    # the edge of the trabecular compartment.
    calculate_trabecular_spacing=True,
    ignore_trabfill_boundary_for_spacing=True,

    # -------------------------------------------------------------------------
    # Structure tensor settings
    # -------------------------------------------------------------------------
    # st_sigma controls the amount of Gaussian smoothing applied before 
    # calculating the structure tensor anisotropy (DA_ST). default is 0.10mm
    use_structure_tensor=True,
    st_sigma=0.10,

    # -------------------------------------------------------------------------
    # Optional Mean Intercept Length (MIL) anisotropy settings
    # -------------------------------------------------------------------------
    # This is slower than the structure-tensor method but is the standard across the field
    # It adds separate MIL columns and does not replace the existing structure-tensor outputs.
    use_mil_anisotropy=True,

    # MIL mode. "standard" is the original Trabmap implementation.
    # "medtool" uses settings and fabric fitting chosen to mimic Medtool's
    # default MIL fabric analysis as closely as possible.
    mil_method="medtool",       # "standard" or "medtool"

    # Standard Trabmap MIL settings.
    mil_n_directions=91,
    mil_line_spacing=None,       # None = one voxel width in mm, unless mil_length_units="voxels"
    mil_sample_step=None,        # None = half a voxel width in mm, unless mil_length_units="voxels"
    mil_min_intersections=2,
    mil_da_mode="bounded",       # "bounded" = 1 - lambda_min/lambda_max; "ratio" = lambda_max/lambda_min

    # Settings to mimic medtools parameters as far as known. 
    # Medtool's -morp defaults are:
    # step=5 voxels, power=2 -> 8*4^2 = 128 directions, validLength=0.
    # Medtool's -faba default is ellipsoid, -fabm default is MIL, and -fabn
    # default is determinant normalization.
    mil_power=None,              # if set, overrides mil_n_directions using 8*4^power
    mil_length_units="mm",       # "mm" or "voxels" for mil_line_spacing/sample_step/valid_length
    mil_valid_length=0.0,        # minimum valid bone intercept/chord length
    mil_fabric_shape="ellipsoid",# "ellipsoid" or "tensor2"
    mil_norm_type="det",         # "det", "trace", or "none"

    # -------------------------------------------------------------------------
    # Optional crop settings
    # -------------------------------------------------------------------------
    # old code should probably not be used, but may be useful in very rare
    # cases 
    use_crop=False,
    crop_z=(0, 300),
    crop_y=(0, 300),
    crop_x=(0, 350),

    # -------------------------------------------------------------------------
    # Output settings
    # -------------------------------------------------------------------------
    save_auto_filled_mask=True,
    save_point_vtp=True,
    save_dense_nifti=False,
    dense_map_mode="local_sphere",  # "local_sphere" or "center_only"
    progress_every=50,

    # -------------------------------------------------------------------------
    # Optional decimated surface mesh export
    # -------------------------------------------------------------------------
    # These surface meshes are intended for downstream analyses and visualization.
    # They are generated from the outer filled-bone stack (_FILL) and the internal
    # trabecular compartment stack (_TRABFILL), then smoothed and decimated to
    # keep file sizes manageable.
    make_surface_meshes=True,
    surface_mesh_suffixes=["_FILL", "_TRABFILL"],
    surface_mesh_output_dir_name="surface_meshes",
    # Primary surface mesh format. VTP is recommended because it preserves
    # Pyvista/Paraview-friendly mesh metadata, and thez are useful for 
    # visualization in paraview If surface_mesh_save_stl=True,
    # a companion STL geometry file is also written for each surface.
    surface_mesh_file_format="vtp",
    surface_mesh_save_stl=True,
    surface_mesh_pad_width=1,
    surface_mesh_keep_largest_component=True,
    surface_mesh_smooth_iterations=30,
    surface_mesh_smooth_relaxation=0.05,
    surface_mesh_decimate=True,
    surface_mesh_target_reduction=0.90,
    surface_mesh_max_file_size_mb=5.0,
    surface_mesh_fail_on_error=True,

    # -------------------------------------------------------------------------
    # Optional tetrahedral mesh export
    # -------------------------------------------------------------------------
    make_mesh=True,

    # Usually leave as None. The batch runner will save:
    # <object>_trabecular_tetra_mesh.vtk inside the object's output folder.
    mesh_out=None,

    # Map CSV values to mesh cells or points using inverse-distance weighting or nearest neighbor.
    mesh_target="cells",      # "cells" or "points"
    mesh_method="idw",        # "idw" or "nearest"
    mesh_k=4,
    mesh_power=2.0,

    # Comma-separated string of columns to map, or None for all numeric non-coordinate columns.
    # Example: "BVTV,EDTThickness_mean,DA_ST". import vlues from config yaml file
    mesh_value_columns=None,

    mesh_pad_width=20,
    mesh_largest_component=True,
    meshfix=True,
    pyacvd=True,
    pyacvd_n_surf=7500, # default was 15000, more likely to fail so set to 7500, this produces a slightly less detail surface mesh
    pyacvd_subdivide=0,
    tetgen_switches="pq1.2Q", # original value, likely results in many vertices on outside fewer inside but smaller file size
    #tetgen_switches="pq1.2a0.05Q" # .05Q constrains inside vertex size, making them more equal to the surface of the mesh, reduce to make intenral even smaller
    # you should probablz use cgal instead anyway
)


def validate_settings(args):
    """Catch common settings mistakes before long computations start"""
    if args.center_inclusion_mode not in ["skip_outside", "zero_outside"]:
        raise ValueError('center_inclusion_mode must be "skip_outside" or "zero_outside".')
    if args.dense_map_mode not in ["local_sphere", "center_only"]:
        raise ValueError('dense_map_mode must be "local_sphere" or "center_only".')
    if args.mesh_target not in ["cells", "points"]:
        raise ValueError('mesh_target must be "cells" or "points".')
    if args.mesh_method not in ["idw", "nearest"]:
        raise ValueError('mesh_method must be "idw" or "nearest".')
    if getattr(args, "mil_method", "standard") not in ["standard", "medtool"]:
        raise ValueError('mil_method must be "standard" or "medtool".')
    if getattr(args, "mil_da_mode", "bounded") not in ["bounded", "ratio"]:
        raise ValueError('mil_da_mode must be "bounded" or "ratio".')
    if getattr(args, "mil_length_units", "mm") not in ["mm", "voxels"]:
        raise ValueError('mil_length_units must be "mm" or "voxels".')
    if getattr(args, "mil_fabric_shape", "ellipsoid") not in ["ellipsoid", "tensor2"]:
        raise ValueError('mil_fabric_shape must be "ellipsoid" or "tensor2".')
    if getattr(args, "mil_norm_type", "det") not in ["det", "trace", "none"]:
        raise ValueError('mil_norm_type must be "det", "trace", or "none".')
    if int(getattr(args, "mil_n_directions", 31)) < 6:
        raise ValueError("mil_n_directions must be at least 6.")
    if getattr(args, "mil_power", None) not in [None, "", "None"]:
        if int(getattr(args, "mil_power")) < 0:
            raise ValueError("mil_power must be >= 0.")
    if getattr(args, "surface_mesh_file_format", "vtp") not in ["vtp", "ply", "stl"]:
        raise ValueError('surface_mesh_file_format must be "vtp", "ply", or "stl".')
    red = float(getattr(args, "surface_mesh_target_reduction", 0.90))
    if not (0.0 <= red < 1.0):
        raise ValueError("surface_mesh_target_reduction must be between 0.0 and 1.0.")


def prefixed_name(args, filename: str) -> str:
    """Return filename with args.output_prefix added, if present."""
    return f"{getattr(args, 'output_prefix', '')}{filename}"


# =============================================================================
# General helpers
# =============================================================================

# check if runner file is open, else it cannot save parameters
def check_runner_csv_writable(csv_path):
    try:
        with open(csv_path, "a"):
            pass
    except PermissionError:
        raise PermissionError(
            f"\nERROR:\n"
            f"The runner CSV appears to be open in Excel:\n\n"
            f"{csv_path}\n\n"
            f"Please close the file and run the script again."
        )


def unit_vector(v: np.ndarray) -> Optional[np.ndarray]:
    n = np.linalg.norm(v)
    return None if n == 0 else v / n


def enforce_sign_convention(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    if np.any(np.isnan(v)):
        return v
    idx = np.argmax(np.abs(v))
    if v[idx] < 0:
        v *= -1.0
    return v


def load_binary_volume(input_path: str) -> np.ndarray:
    """Load a binary 3D volume from a folder of TIFF slices, a 3D TIFF, or .npy."""
    if input_path is None:
        raise ValueError("Input path is None.")

    if os.path.isdir(input_path):
        files = sorted(
            glob.glob(os.path.join(input_path, "*.tif"))
            + glob.glob(os.path.join(input_path, "*.tiff"))
        )
        if len(files) == 0:
            raise ValueError(f"No tif/tiff files found in folder: {input_path}")
        arr = tifffile.imread(files)
    else:
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".npy":
            arr = np.load(input_path)
        elif ext in [".tif", ".tiff"]:
            arr = tifffile.imread(input_path)
        else:
            raise ValueError("Input must be .npy, .tif/.tiff, or a folder of tif slices.")

    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[-1] in [1, 3, 4]:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unsupported 4D array shape: {arr.shape}")
    if arr.ndim != 3:
        raise ValueError(f"Loaded array must be 3D after preprocessing, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def load_optional_mask(mask_path: Optional[str], expected_shape: tuple) -> Optional[np.ndarray]:
    if mask_path is None or str(mask_path).lower() in ["none", ""]:
        return None
    mask = load_binary_volume(mask_path)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {expected_shape}")
    return mask.astype(bool)


def fill_mask_multiaxial(binary_vol: np.ndarray) -> np.ndarray:
    """Create a filled-bone mask by filling holes slice-by-slice in all three axes."""
    binary_vol = binary_vol > 0
    filled_z = np.zeros_like(binary_vol, dtype=bool)
    filled_y = np.zeros_like(binary_vol, dtype=bool)
    filled_x = np.zeros_like(binary_vol, dtype=bool)

    for z in range(binary_vol.shape[0]):
        filled_z[z] = ndimage.binary_fill_holes(binary_vol[z])
    for y in range(binary_vol.shape[1]):
        filled_y[:, y, :] = ndimage.binary_fill_holes(binary_vol[:, y, :])
    for x in range(binary_vol.shape[2]):
        filled_x[:, :, x] = ndimage.binary_fill_holes(binary_vol[:, :, x])

    filled = filled_z | filled_y | filled_x
    filled |= binary_vol
    return filled


def load_landmarks_csv(csv_path: str) -> dict[str, np.ndarray]:
    landmarks = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            landmarks[row["name"]] = np.array(
                [float(row["z"]), float(row["y"]), float(row["x"])], dtype=float
            )
    required = {"origin", "long_axis", "rotation_ref"}
    missing = required - set(landmarks.keys())
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")
    return landmarks


def build_anatomical_basis_from_landmarks(landmarks: dict[str, np.ndarray], spacing: tuple) -> np.ndarray:
    origin = landmarks["origin"] * np.array(spacing)
    p_long = landmarks["long_axis"] * np.array(spacing)
    p_rot = landmarks["rotation_ref"] * np.array(spacing)

    axis1 = unit_vector(p_long - origin)
    if axis1 is None:
        raise ValueError("origin and long_axis are identical or invalid.")
    tmp = unit_vector(p_rot - origin)
    if tmp is None:
        raise ValueError("origin and rotation_ref are identical or invalid.")
    axis3 = unit_vector(np.cross(axis1, tmp))
    if axis3 is None:
        raise ValueError("Landmarks are collinear; cannot define a rotational frame.")
    axis2 = unit_vector(np.cross(axis3, axis1))
    return np.column_stack([axis1, axis2, axis3])


def transform_vector_to_anatomical(v_image: np.ndarray, anat_basis: np.ndarray) -> np.ndarray:
    if v_image is None:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    v_image = np.asarray(v_image, dtype=float)
    if np.any(np.isnan(v_image)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return anat_basis.T @ v_image


def xyz_pixels_to_zyx_physical(point_xyz, spacing: tuple) -> np.ndarray:
    """Convert x,y,z pixel coordinates from the runner CSV to z,y,x physical coordinates."""
    point_xyz = np.asarray(point_xyz, dtype=float)
    if point_xyz.shape != (3,):
        raise ValueError("Landmark point must contain exactly three values: x,y,z.")
    if np.any(~np.isfinite(point_xyz)):
        raise ValueError("Landmark point contains non-finite values.")
    return np.array(
        [
            point_xyz[2] * spacing[0],  # z pixel * z spacing
            point_xyz[1] * spacing[1],  # y pixel * y spacing
            point_xyz[0] * spacing[2],  # x pixel * x spacing
        ],
        dtype=float,
    )


def build_custom_anatomical_frame(args, spacing: tuple):
    """Build a custom anatomical basis from runner CSV landmarks.

    The runner CSV stores landmarks as x,y,z pixel coordinates. Internally the
    image arrays use z,y,x order, so this function converts x,y,z pixels to
    z,y,x physical coordinates before calculating the frame.

    Returns the following
    -------
    tuple
        (anat_basis, long_axis), where anat_basis columns are axis1, axis2,
        axis3 in image z,y,x coordinates. If no complete custom frame is active,
        returns (None, None).
    """
    if not getattr(args, "use_custom_anatomical_frame", False):
        return None, None

    p1_xyz = getattr(args, "custom_long_axis_p1_xyz", None)
    p2_xyz = getattr(args, "custom_long_axis_p2_xyz", None)
    prot_xyz = getattr(args, "custom_rotation_point_xyz", None)
    if p1_xyz is None or p2_xyz is None or prot_xyz is None:
        return None, None

    p1 = xyz_pixels_to_zyx_physical(p1_xyz, spacing)
    p2 = xyz_pixels_to_zyx_physical(p2_xyz, spacing)
    prot = xyz_pixels_to_zyx_physical(prot_xyz, spacing)

    axis1 = unit_vector(p2 - p1)
    if axis1 is None:
        raise ValueError("Custom long-axis landmarks are identical; cannot define anatomical axis 1.")

    tmp = unit_vector(prot - p1)
    if tmp is None:
        raise ValueError("Custom rotation landmark is identical to landmark_long_1; cannot define rotation around axis 1.")

    axis3 = unit_vector(np.cross(axis1, tmp))
    if axis3 is None:
        raise ValueError(
            "Custom landmarks are collinear; the rotation landmark must not lie exactly on the custom long axis."
        )

    axis2 = unit_vector(np.cross(axis3, axis1))
    anat_basis = np.column_stack([axis1, axis2, axis3])
    return anat_basis, enforce_sign_convention(axis1)


def build_custom_long_axis_vector(args, spacing: tuple) -> Optional[np.ndarray]:
    """Backward-compatible helper: return only the custom long-axis vector."""
    _, long_axis = build_custom_anatomical_frame(args, spacing)
    return long_axis

def angle_to_axis_degrees(v_image: np.ndarray, axis_image: Optional[np.ndarray]) -> float:
    """Return the undirected angle between a vector and an axis, in degrees.

    Because trabecular orientation is axial rather than directional, v and -v are
    treated as equivalent. Output is therefore constrained to 0-90 degrees.
    in other words, a trabec poiting up is the same as a trabec pointing down
    this prevents arbitrary differences in trabecular direction
    """
    if axis_image is None:
        return np.nan
    v = unit_vector(np.asarray(v_image, dtype=float))
    a = unit_vector(np.asarray(axis_image, dtype=float))
    if v is None or a is None:
        return np.nan
    if np.any(np.isnan(v)) or np.any(np.isnan(a)):
        return np.nan
    cosang = float(np.clip(abs(np.dot(v, a)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def fibonacci_hemisphere_directions(n: int) -> np.ndarray:
    """Return approximately uniform undirected test directions on one hemisphere.

    MIL is axial, so d and -d are equivalent. Directions are returned in image
    z,y,x order and normalized to unit length.
    """
    n = int(n)
    if n < 1:
        return np.empty((0, 3), dtype=float)
    dirs = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    # Generate more full-sphere candidates, then fold into one hemisphere and
    # remove near-duplicates by sign convention.
    for i in range(max(2 * n, 12)):
        z = 1.0 - 2.0 * (i + 0.5) / max(2 * n, 12)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        v = np.array([z, r * np.sin(theta), r * np.cos(theta)], dtype=float)
        v = enforce_sign_convention(v)
        v = unit_vector(v)
        if v is not None:
            dirs.append(v)
    unique = []
    for v in dirs:
        if all(abs(float(np.dot(v, u))) < 0.999 for u in unique):
            unique.append(v)
        if len(unique) >= n:
            break
    return np.asarray(unique, dtype=float)


def perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to direction."""
    d = unit_vector(direction)
    if d is None:
        raise ValueError("Cannot create perpendicular basis for zero vector.")
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(d, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = unit_vector(np.cross(d, ref))
    if u is None:
        ref = np.array([0.0, 0.0, 1.0])
        u = unit_vector(np.cross(d, ref))
    v = unit_vector(np.cross(d, u))
    return u, v


def mil_for_direction_in_sphere(
    bone: np.ndarray,
    center: tuple,
    radius_phys: float,
    spacing: tuple,
    direction: np.ndarray,
    line_spacing: float,
    sample_step: float,
    min_intersections: int = 2,
) -> tuple[float, int, float]:
    """Original Trabmap MIL estimate for one direction within a spherical ROI.

    This legacy mode counts binary transitions along parallel test lines and
    returns total sampled test-line length divided by the number of transitions.
    It is kept for backwards compatibility. For Medtool-like behavior, use
    ``mil_for_direction_in_sphere_medtool``.
    """
    d = unit_vector(np.asarray(direction, dtype=float))
    if d is None:
        return np.nan, 0, 0.0
    u, v = perpendicular_basis(d)

    center_phys = np.array(
        [center[0] * spacing[0], center[1] * spacing[1], center[2] * spacing[2]],
        dtype=float,
    )
    offsets = np.arange(-radius_phys, radius_phys + 1e-9, line_spacing, dtype=float)
    t_vals = np.arange(-radius_phys, radius_phys + 1e-9, sample_step, dtype=float)

    total_length = 0.0
    total_intersections = 0

    for a in offsets:
        for b in offsets:
            offset2 = a * a + b * b
            if offset2 > radius_phys * radius_phys:
                continue
            half_len = float(np.sqrt(max(radius_phys * radius_phys - offset2, 0.0)))
            use_t = t_vals[(t_vals >= -half_len) & (t_vals <= half_len)]
            if use_t.size < 2:
                continue

            pts_phys = center_phys + a * u + b * v + use_t[:, None] * d
            coords = np.column_stack(
                [
                    np.rint(pts_phys[:, 0] / spacing[0]).astype(int),
                    np.rint(pts_phys[:, 1] / spacing[1]).astype(int),
                    np.rint(pts_phys[:, 2] / spacing[2]).astype(int),
                ]
            )
            valid = (
                (coords[:, 0] >= 0) & (coords[:, 0] < bone.shape[0]) &
                (coords[:, 1] >= 0) & (coords[:, 1] < bone.shape[1]) &
                (coords[:, 2] >= 0) & (coords[:, 2] < bone.shape[2])
            )
            if np.count_nonzero(valid) < 2:
                continue
            vals = np.zeros(coords.shape[0], dtype=np.uint8)
            cvalid = coords[valid]
            vals[valid] = bone[cvalid[:, 0], cvalid[:, 1], cvalid[:, 2]] > 0
            transitions = int(np.count_nonzero(vals[1:] != vals[:-1]))
            if transitions > 0:
                total_intersections += transitions
            total_length += float((use_t.size - 1) * sample_step)

    if total_intersections < int(min_intersections) or total_length <= 0:
        return np.nan, total_intersections, total_length
    return float(total_length / total_intersections), total_intersections, total_length


def _binary_run_lengths(vals: np.ndarray, sample_step: float) -> list[float]:
    """Return physical lengths of contiguous True/bone runs along one sampled line."""
    vals = np.asarray(vals, dtype=bool)
    lengths = []
    i = 0
    n = len(vals)
    while i < n:
        if not vals[i]:
            i += 1
            continue
        j = i
        while j < n and vals[j]:
            j += 1
        # If a run contains k sampled points, its chord length is approximated as
        # k * sample_step. This is because I believe Medtool's algorithm operates on sampled rays in voxel units.
        lengths.append(float((j - i) * sample_step))
        i = j
    return lengths


def mil_for_direction_in_sphere_medtool(
    bone: np.ndarray,
    center: tuple,
    radius_phys: float,
    spacing: tuple,
    direction: np.ndarray,
    line_spacing: float,
    sample_step: float,
    valid_length: float = 0.0,
) -> tuple[float, int, float]:
    """Medtool-like MIL settings for one direction within a spherical ROI.

    Medtool's morphology options define a ray-field spacing (``step``), number
    of directions, and minimum valid intercept length (``validLength``). This
    function therefore estimates MIL as the mean length of valid bone chords
    intercepted by a parallel ray field. Chords shorter than ``valid_length`` are
    ignored. ``valid_length=0`` reproduces Medtool's default.

    This is not going to be identical to Medtool because the exact
    proprietary ray generation and ellipsoid fitting details are not available,
    but it is closer than the previous implementation
    """
    d = unit_vector(np.asarray(direction, dtype=float))
    if d is None:
        return np.nan, 0, 0.0
    u, v = perpendicular_basis(d)

    center_phys = np.array(
        [center[0] * spacing[0], center[1] * spacing[1], center[2] * spacing[2]],
        dtype=float,
    )
    offsets = np.arange(-radius_phys, radius_phys + 1e-9, line_spacing, dtype=float)
    t_vals = np.arange(-radius_phys, radius_phys + 1e-9, sample_step, dtype=float)

    valid_chord_lengths = []
    total_sampled_length = 0.0

    for a in offsets:
        for b in offsets:
            offset2 = a * a + b * b
            if offset2 > radius_phys * radius_phys:
                continue
            half_len = float(np.sqrt(max(radius_phys * radius_phys - offset2, 0.0)))
            use_t = t_vals[(t_vals >= -half_len) & (t_vals <= half_len)]
            if use_t.size < 2:
                continue

            pts_phys = center_phys + a * u + b * v + use_t[:, None] * d
            coords = np.column_stack(
                [
                    np.rint(pts_phys[:, 0] / spacing[0]).astype(int),
                    np.rint(pts_phys[:, 1] / spacing[1]).astype(int),
                    np.rint(pts_phys[:, 2] / spacing[2]).astype(int),
                ]
            )
            valid = (
                (coords[:, 0] >= 0) & (coords[:, 0] < bone.shape[0]) &
                (coords[:, 1] >= 0) & (coords[:, 1] < bone.shape[1]) &
                (coords[:, 2] >= 0) & (coords[:, 2] < bone.shape[2])
            )
            if np.count_nonzero(valid) < 2:
                continue

            vals = np.zeros(coords.shape[0], dtype=np.uint8)
            cvalid = coords[valid]
            vals[valid] = bone[cvalid[:, 0], cvalid[:, 1], cvalid[:, 2]] > 0

            for length in _binary_run_lengths(vals, sample_step):
                if length >= float(valid_length):
                    valid_chord_lengths.append(float(length))

            total_sampled_length += float((use_t.size - 1) * sample_step)

    n_chords = len(valid_chord_lengths)
    if n_chords == 0:
        return np.nan, 0, total_sampled_length
    return float(np.mean(valid_chord_lengths)), n_chords, total_sampled_length


def _normalize_fabric_eigenvalues(eig: np.ndarray, norm_type: str) -> np.ndarray:
    """Normalize fabric eigenvalues"""
    eig = np.asarray(eig, dtype=float).copy()
    if np.any(~np.isfinite(eig)) or np.any(eig <= 0):
        return eig
    norm_type = str(norm_type).lower()
    if norm_type == "trace":
        s = float(np.sum(eig))
        return eig * (3.0 / s) if s > 0 else eig
    if norm_type == "det":
        det = float(np.prod(eig))
        return eig / (det ** (1.0 / 3.0)) if det > 0 else eig
    return eig


def _ellipsoid_fabric_from_mil(directions: np.ndarray, mils: np.ndarray, norm_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit an ellipsoid to directional MIL values.

    fit the quadratic form d.T A d = 1 / MIL(d)^2. The ellipsoid radii are
    then 1/sqrt(eigenvalues(A)). These radii are used as fabric eigenvalues,
    sorted from smallest to largest. The eigenvector associated with the largest
    radius is the principal MIL direction.
    """
    D = np.asarray(directions, dtype=float)
    M = np.asarray(mils, dtype=float)
    ok = np.isfinite(M) & (M > 0)
    D = D[ok]
    M = M[ok]
    if len(M) < 6:
        return np.array([np.nan, np.nan, np.nan], dtype=float), np.full((3, 3), np.nan, dtype=float)

    x, y, z = D[:, 0], D[:, 1], D[:, 2]
    X = np.column_stack([x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z])
    b = 1.0 / (M * M)

    try:
        coeff, *_ = np.linalg.lstsq(X, b, rcond=None)
        A = np.array(
            [
                [coeff[0], coeff[3], coeff[4]],
                [coeff[3], coeff[1], coeff[5]],
                [coeff[4], coeff[5], coeff[2]],
            ],
            dtype=float,
        )
        eig_a, vecs_a = np.linalg.eigh(A)
        if np.any(~np.isfinite(eig_a)) or np.any(eig_a <= 0):
            raise ValueError("Non-positive ellipsoid eigenvalue")
        radii = 1.0 / np.sqrt(eig_a)
        order = np.argsort(radii)
        radii = radii[order]
        vecs = vecs_a[:, order]
        radii = _normalize_fabric_eigenvalues(radii, norm_type=norm_type)
        return radii, vecs
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=float), np.full((3, 3), np.nan, dtype=float)


def _tensor2_fabric_from_mil(directions: np.ndarray, mils: np.ndarray, norm_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Second-order tensor approximation from directional MIL values."""
    D = np.asarray(directions, dtype=float)
    M = np.asarray(mils, dtype=float)
    ok = np.isfinite(M) & (M > 0)
    D = D[ok]
    M = M[ok]
    if len(M) < 6:
        return np.array([np.nan, np.nan, np.nan], dtype=float), np.full((3, 3), np.nan, dtype=float)

    weights = M ** 2
    fabric = np.zeros((3, 3), dtype=float)
    for w, d in zip(weights, D):
        fabric += w * np.outer(d, d)
    fabric /= float(np.sum(weights))
    eig, vecs = np.linalg.eigh(fabric)
    order = np.argsort(eig)
    eig = eig[order]
    vecs = vecs[:, order]
    eig = _normalize_fabric_eigenvalues(eig, norm_type=norm_type)
    return eig, vecs


def local_mil_fabric(
    bone: np.ndarray,
    center: tuple,
    radius_phys: float,
    spacing: tuple,
    directions: np.ndarray,
    line_spacing: float,
    sample_step: float,
    min_intersections: int = 2,
    da_mode: str = "bounded",
    method: str = "standard",
    valid_length: float = 0.0,
    fabric_shape: str = "ellipsoid",
    norm_type: str = "det",
) -> dict[str, object]:
    """Estimate local MIL fabric tensor and anisotropy for one spherical ROI.

    method="standard" keeps the older Trabmap transition-count approach.
    method="medtool" uses a chord-length ray-field approach with ellipsoid
    fitting and determinant normalization, matching Medtool defaults as closely
    as possible from public descriptions. Still not exactly the same as 
    medtool due to missing information on the exact algorithm but closer than before
    """
    mils = []
    valid_dirs = []
    total_intersections = 0
    total_line_length = 0.0
    method = str(method).lower()

    for d in directions:
        if method == "medtool":
            mil, nint, llen = mil_for_direction_in_sphere_medtool(
                bone=bone,
                center=center,
                radius_phys=radius_phys,
                spacing=spacing,
                direction=d,
                line_spacing=line_spacing,
                sample_step=sample_step,
                valid_length=valid_length,
            )
        else:
            mil, nint, llen = mil_for_direction_in_sphere(
                bone=bone,
                center=center,
                radius_phys=radius_phys,
                spacing=spacing,
                direction=d,
                line_spacing=line_spacing,
                sample_step=sample_step,
                min_intersections=min_intersections,
            )
        if np.isfinite(mil) and mil > 0:
            mils.append(float(mil))
            valid_dirs.append(np.asarray(d, dtype=float))
        total_intersections += int(nint)
        total_line_length += float(llen)

    if len(mils) < 6:
        return {
            "eig": np.array([np.nan, np.nan, np.nan], dtype=float),
            "dir": np.array([np.nan, np.nan, np.nan], dtype=float),
            "anisotropy": np.nan,
            "valid_directions": len(mils),
            "mean_mil": np.nan,
            "total_intersections": total_intersections,
            "total_line_length": total_line_length,
        }

    mils = np.asarray(mils, dtype=float)
    valid_dirs = np.asarray(valid_dirs, dtype=float)

    if method == "medtool" and str(fabric_shape).lower() == "ellipsoid":
        eig, vecs = _ellipsoid_fabric_from_mil(valid_dirs, mils, norm_type=norm_type)
    else:
        eig, vecs = _tensor2_fabric_from_mil(valid_dirs, mils, norm_type=norm_type)

    if np.any(~np.isfinite(eig)) or vecs.shape != (3, 3):
        principal = np.array([np.nan, np.nan, np.nan], dtype=float)
        anisotropy = np.nan
    else:
        principal = enforce_sign_convention(vecs[:, -1])
        if eig[-1] > 0 and eig[0] > 0:
            if da_mode == "ratio":
                anisotropy = float(eig[-1] / eig[0])
            else:
                anisotropy = float(1.0 - eig[0] / eig[-1])
        else:
            anisotropy = np.nan

    return {
        "eig": eig,
        "dir": principal,
        "anisotropy": anisotropy,
        "valid_directions": len(mils),
        "mean_mil": float(np.mean(mils)),
        "total_intersections": total_intersections,
        "total_line_length": total_line_length,
    }


# =============================================================================
# Sphere analysis
# =============================================================================

def generate_grid_centers(shape: tuple, spacing: tuple, sphere_radius: float, grid_step: float, margin=None):
    if margin is None:
        margin = sphere_radius
    z_extent = (shape[0] - 1) * spacing[0]
    y_extent = (shape[1] - 1) * spacing[1]
    x_extent = (shape[2] - 1) * spacing[2]
    z_vals = np.arange(margin, z_extent - margin + 1e-9, grid_step)
    y_vals = np.arange(margin, y_extent - margin + 1e-9, grid_step)
    x_vals = np.arange(margin, x_extent - margin + 1e-9, grid_step)
    centers = []
    for z in z_vals:
        for y in y_vals:
            for x in x_vals:
                centers.append((int(round(z / spacing[0])), int(round(y / spacing[1])), int(round(x / spacing[2]))))
    return centers


def center_is_valid(center: tuple, filled_mask: Optional[np.ndarray]) -> bool:
    return True if filled_mask is None else bool(filled_mask[center])


def precompute_sphere_offsets(radius_phys: float, spacing: tuple) -> np.ndarray:
    rz = int(np.ceil(radius_phys / spacing[0]))
    ry = int(np.ceil(radius_phys / spacing[1]))
    rx = int(np.ceil(radius_phys / spacing[2]))
    offsets = []
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                dist2 = (dz * spacing[0]) ** 2 + (dy * spacing[1]) ** 2 + (dx * spacing[2]) ** 2
                if dist2 <= radius_phys ** 2:
                    offsets.append((dz, dy, dx))
    return np.array(offsets, dtype=np.int32)


def get_sphere_coords(center: tuple, offsets: np.ndarray, shape: tuple):
    coords = offsets + np.array(center, dtype=np.int32)
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < shape[2])
    )
    coords = coords[valid]
    return coords[:, 0], coords[:, 1], coords[:, 2]


def bvtv_in_sphere_using_filled_mask(z, y, x, bone, filled_mask=None):
    bone_vals = bone[z, y, x] > 0
    if filled_mask is not None:
        denom_mask = filled_mask[z, y, x] > 0
        denom = int(np.count_nonzero(denom_mask))
        if denom == 0:
            return np.nan
        num = int(np.count_nonzero(bone_vals & denom_mask))
        return num / denom
    return np.nan if bone_vals.size == 0 else float(np.mean(bone_vals))


def structure_tensor_components_3d(image: np.ndarray, sigma: float):
    elems = structure_tensor(image, sigma=sigma, order="rc")
    if len(elems) != 6:
        raise ValueError(f"Expected 6 structure-tensor elements for 3D, got {len(elems)}")
    return elems[0], elems[1], elems[2], elems[3], elems[4], elems[5]


def make_affine(spacing):
    affine = np.eye(4, dtype=float)
    affine[0, 0] = spacing[2]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[0]
    return affine


def save_nifti(array: np.ndarray, spacing: tuple, out_path: str):
    if nib is None:
        raise ImportError("nibabel is required to save NIfTI. Install with: pip install nibabel")
    nib.save(nib.Nifti1Image(array.astype(np.float32), make_affine(spacing)), out_path)


def center_only_heatmap(shape: tuple, centers: list, values: list) -> np.ndarray:
    sum_vol = np.zeros(shape, dtype=np.float32)
    cnt_vol = np.zeros(shape, dtype=np.float32)
    for center, value in zip(centers, values):
        if np.isnan(value):
            continue
        z, y, x = center
        sum_vol[z, y, x] += np.float32(value)
        cnt_vol[z, y, x] += 1.0
    out = np.full(shape, np.nan, dtype=np.float32)
    valid = cnt_vol > 0
    out[valid] = sum_vol[valid] / cnt_vol[valid]
    return out


def local_sphere_heatmap(shape: tuple, centers: list, values: list, sphere_radius: float, spacing: tuple) -> np.ndarray:
    sum_vol = np.zeros(shape, dtype=np.float32)
    cnt_vol = np.zeros(shape, dtype=np.float32)
    rz = int(np.ceil(sphere_radius / spacing[0]))
    ry = int(np.ceil(sphere_radius / spacing[1]))
    rx = int(np.ceil(sphere_radius / spacing[2]))
    for center, value in zip(centers, values):
        if np.isnan(value):
            continue
        cz, cy, cx = center
        z0, z1 = max(0, cz - rz), min(shape[0], cz + rz + 1)
        y0, y1 = max(0, cy - ry), min(shape[1], cy + ry + 1)
        x0, x1 = max(0, cx - rx), min(shape[2], cx + rx + 1)
        zz, yy, xx = np.indices((z1 - z0, y1 - y0, x1 - x0))
        zz, yy, xx = zz + z0, yy + y0, xx + x0
        dz = (zz - cz) * spacing[0]
        dy = (yy - cy) * spacing[1]
        dx = (xx - cx) * spacing[2]
        mask = (dz ** 2 + dy ** 2 + dx ** 2) <= sphere_radius ** 2
        sum_vol[z0:z1, y0:y1, x0:x1][mask] += np.float32(value)
        cnt_vol[z0:z1, y0:y1, x0:x1][mask] += 1.0
    out = np.full(shape, np.nan, dtype=np.float32)
    valid = cnt_vol > 0
    out[valid] = sum_vol[valid] / cnt_vol[valid]
    return out


def export_spheres_to_vtp(results_df: pd.DataFrame, out_path: str, use_only_inside=True):
    if pv is None:
        raise ImportError("pyvista is required to save VTP. Install with: pip install pyvista")
    df = results_df.copy()
    if use_only_inside and "inside_filled_mask" in df.columns:
        df = df[df["inside_filled_mask"] == 1].copy()

    points = np.vstack([
        df["center_x_phys"].to_numpy(),
        df["center_y_phys"].to_numpy(),
        df["center_z_phys"].to_numpy(),
    ]).T
    cloud = pv.PolyData(points)

    scalar_cols = [
        "BVTV", "rBVTV", "BVTV_zscore", "BVTV_minmax",
        "valid_trabfill_voxels_in_sphere", "total_voxels_in_sphere",
        "valid_fraction_trabfill", "distance_to_trabfill_boundary_mm",
        "distance_to_trabfill_boundary_vox",
        "EDTThickness_mean", "EDTThickness_max",
        "EDTSpacing_mean", "EDTSpacing_max",
        "DA_ST", "angle_to_custom_long_axis_deg",
        "DA_MIL", "mean_intercept_length_mil", "mil_valid_directions",
        "mil_total_intersections", "mil_total_line_length",
        "angle_mil_to_custom_long_axis_deg",
    ]
    for col in [c for c in scalar_cols if c in df.columns]:
        cloud[col] = df[col].to_numpy()

    if all(c in df.columns for c in ["primary_dir_st_x", "primary_dir_st_y", "primary_dir_st_z"]):
        vec = np.vstack([
            df["primary_dir_st_x"].to_numpy(),
            df["primary_dir_st_y"].to_numpy(),
            df["primary_dir_st_z"].to_numpy(),
        ]).T
        cloud["principal_direction"] = vec

    if all(c in df.columns for c in ["primary_dir_st_anat_1", "primary_dir_st_anat_2", "primary_dir_st_anat_3"]):
        veca = np.vstack([
            df["primary_dir_st_anat_1"].to_numpy(),
            df["primary_dir_st_anat_2"].to_numpy(),
            df["primary_dir_st_anat_3"].to_numpy(),
        ]).T
        cloud["principal_direction_anatomical"] = veca


    if all(c in df.columns for c in ["primary_dir_mil_x", "primary_dir_mil_y", "primary_dir_mil_z"]):
        vec_mil = np.vstack([
            df["primary_dir_mil_x"].to_numpy(),
            df["primary_dir_mil_y"].to_numpy(),
            df["primary_dir_mil_z"].to_numpy(),
        ]).T
        cloud["principal_direction_mil"] = vec_mil

    if all(c in df.columns for c in ["primary_dir_mil_anat_1", "primary_dir_mil_anat_2", "primary_dir_mil_anat_3"]):
        vec_mil_anat = np.vstack([
            df["primary_dir_mil_anat_1"].to_numpy(),
            df["primary_dir_mil_anat_2"].to_numpy(),
            df["primary_dir_mil_anat_3"].to_numpy(),
        ]).T
        cloud["principal_direction_mil_anatomical"] = vec_mil_anat

    if "DA_ST" in df.columns and "principal_direction" in cloud.array_names:
        cloud["dir_weighted"] = cloud["principal_direction"] * df["DA_ST"].to_numpy()[:, None]

    if "DA_MIL" in df.columns and "principal_direction_mil" in cloud.array_names:
        cloud["dir_weighted_mil"] = cloud["principal_direction_mil"] * df["DA_MIL"].to_numpy()[:, None]

    cloud.save(out_path)
    return out_path


def analyze_spheres(args) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray, tuple]:
    spacing = (args.spacing_z, args.spacing_y, args.spacing_x)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading bone volume...")
    bone = load_binary_volume(args.input)

    filled_mask = load_optional_mask(args.filled_mask, bone.shape) if args.filled_mask else None
    auto_filled_mask_created = False
    if filled_mask is None:
        print("No filled-bone mask provided. Creating filled mask from binary stack...")
        filled_mask = fill_mask_multiaxial(bone)
        auto_filled_mask_created = True
    else:
        print("Using user-provided filled-bone mask.")

    anat_basis = None
    custom_frame_basis, custom_long_axis = build_custom_anatomical_frame(args, spacing)

    if custom_frame_basis is not None:
        # Runner CSV landmarks take priority because they are object-specific.
        anat_basis = custom_frame_basis
        landmarks = None
    elif args.landmarks_csv:
        landmarks = load_landmarks_csv(args.landmarks_csv)
        anat_basis = build_anatomical_basis_from_landmarks(landmarks, spacing)
    else:
        landmarks = None

    if args.use_crop:
        z0, z1 = args.crop_z
        y0, y1 = args.crop_y
        x0, x1 = args.crop_x
        bone = bone[z0:z1, y0:y1, x0:x1]
        filled_mask = filled_mask[z0:z1, y0:y1, x0:x1]

    print("Bone shape:", bone.shape)
    print("Bone voxels:", int(bone.sum()))
    print("Filled-mask voxels:", int(filled_mask.sum()))
    print("Auto filled mask created:", auto_filled_mask_created)
    print("Landmarks CSV present:", landmarks is not None)
    print("Custom anatomical frame present:", custom_frame_basis is not None)
    print("Custom long axis present:", custom_long_axis is not None)

    print("Computing trabecular thickness EDT...")
    # For thickness, this prevent voxels outside TRABFILL from acting as
    # biologically meaningful background. This avoids artificially reducing
    # thickness for trabecular voxels near the TRABFILL boundary.
    if getattr(args, "ignore_trabfill_boundary_for_thickness", True) and filled_mask is not None:
        thickness_edt_mask = (bone > 0) | (~filled_mask.astype(bool))
        edt = ndimage.distance_transform_edt(thickness_edt_mask, sampling=spacing)
        print("Thickness EDT mode: outside TRABFILL ignored as background")
    else:
        edt = ndimage.distance_transform_edt(bone, sampling=spacing)
        print("Thickness EDT mode: standard bone-mask EDT")
    print("EDT min/max:", float(np.min(edt)), float(np.max(edt)))

    spacing_edt = None
    if getattr(args, "calculate_trabecular_spacing", True):
        print("Computing trabecular spacing EDT...")
        # Tb.Sp is estimated from the distance transform of marrow space inside
        # TRABFILL. Optionally, outside-TRABFILL voxels are treated as valid
        # non-background so that the compartment boundary does not artificially
        # reduce spacing values near the boundary.
        if filled_mask is not None:
            marrow_inside_trabfill = filled_mask.astype(bool) & (~(bone > 0))
            if getattr(args, "ignore_trabfill_boundary_for_spacing", True):
                spacing_edt_mask = marrow_inside_trabfill | (~filled_mask.astype(bool))
                print("Spacing EDT mode: outside TRABFILL ignored as background")
            else:
                spacing_edt_mask = marrow_inside_trabfill
                print("Spacing EDT mode: marrow inside TRABFILL only")
        else:
            spacing_edt_mask = ~(bone > 0)
            print("Spacing EDT mode: standard inverse bone mask")
        spacing_edt = ndimage.distance_transform_edt(spacing_edt_mask, sampling=spacing)
        print("Spacing EDT min/max:", float(np.min(spacing_edt)), float(np.max(spacing_edt)))

    filled_boundary_edt = None
    if getattr(args, "save_boundary_diagnostics", True) and filled_mask is not None:
        print("Computing TRABFILL boundary-distance EDT...")
        filled_boundary_edt = ndimage.distance_transform_edt(filled_mask.astype(bool), sampling=spacing)

    st_components = None
    if args.use_structure_tensor:
        print("Computing structure tensor...")
        if getattr(args, "clip_non_bvtv_to_filled_mask", True) and filled_mask is not None:
            st_valid_bone = (bone > 0) & filled_mask.astype(bool)
            print("Structure tensor mode: clipped to trabecular bone inside TRABFILL")
        else:
            st_valid_bone = bone > 0
            print("Structure tensor mode: standard bone mask")
        st_image = edt.astype(np.float32) * st_valid_bone.astype(np.float32)
        sigma_vox = float(args.st_sigma) / min(spacing)
        st_components = structure_tensor_components_3d(
            st_image,
            sigma=sigma_vox
        )
        del st_image
        gc.collect()

        if getattr(args, "save_mapping_debug_images", False):
            debug_dir = Path(args.out_dir).parent / "segmentations" / f"{getattr(args, 'object', 'object')}_DEBUG"

            print("Saving mapping debug images to:", debug_dir)

            save_debug_mid_slices(
                edt,
                debug_dir,
                "DEBUG_mapping_EDT",
                mask=filled_mask
            )

            Jzz, Jzy, Jzx, Jyy, Jyx, Jxx = st_components
            st_trace = Jzz + Jyy + Jxx

            save_debug_mid_slices(
                st_trace,
                debug_dir,
                "DEBUG_mapping_ST_trace",
                mask=filled_mask
            )

            if spacing_edt is not None:
                save_debug_mid_slices(
                    spacing_edt,
                    debug_dir,
                    "DEBUG_mapping_spacing_EDT",
                    mask=filled_mask
                )

            del st_trace
            gc.collect()

        for i, comp in enumerate(st_components, start=1):
            print(f"Component {i}: shape={comp.shape}, min={float(np.nanmin(comp)):.6g}, max={float(np.nanmax(comp)):.6g}")
    else:
        print("Structure tensor disabled.")

    centers_all = generate_grid_centers(bone.shape, spacing, args.sphere_radius, args.grid_step, args.margin)
    inside_flags = [center_is_valid(c, filled_mask) for c in centers_all]
    print("Total grid centers:", len(centers_all))
    print("Centers inside filled mask:", int(np.sum(inside_flags)))
    print("Centers outside filled mask:", int(len(centers_all) - np.sum(inside_flags)))

    sphere_offsets = precompute_sphere_offsets(args.sphere_radius, spacing)
    print("Number of sphere offsets:", sphere_offsets.shape[0])

    mil_directions = None
    mil_line_spacing = None
    mil_sample_step = None
    mil_valid_length = 0.0
    if getattr(args, "use_mil_anisotropy", False):
        mil_power = getattr(args, "mil_power", None)
        if mil_power not in [None, "", "None"]:
            mil_n_directions = int(8 * (4 ** int(mil_power)))
        else:
            mil_n_directions = int(getattr(args, "mil_n_directions", 31))
        mil_directions = fibonacci_hemisphere_directions(mil_n_directions)

        min_spacing = float(min(spacing))
        mil_length_units = str(getattr(args, "mil_length_units", "mm")).lower()

        raw_line_spacing = getattr(args, "mil_line_spacing", None)
        raw_sample_step = getattr(args, "mil_sample_step", None)
        raw_valid_length = getattr(args, "mil_valid_length", 0.0)

        if raw_line_spacing in [None, "", "None"]:
            raw_line_spacing = 5.0 if str(getattr(args, "mil_method", "standard")).lower() == "medtool" else 1.0
        if raw_sample_step in [None, "", "None"]:
            raw_sample_step = raw_line_spacing
        if raw_valid_length in [None, "", "None"]:
            raw_valid_length = 0.0

        if mil_length_units == "voxels":
            mil_line_spacing = float(raw_line_spacing) * min_spacing
            mil_sample_step = float(raw_sample_step) * min_spacing
            mil_valid_length = float(raw_valid_length) * min_spacing
        else:
            mil_line_spacing = float(raw_line_spacing)
            mil_sample_step = float(raw_sample_step)
            mil_valid_length = float(raw_valid_length)

        print(
            "MIL anisotropy enabled: "
            f"method={getattr(args, 'mil_method', 'standard')}, "
            f"{len(mil_directions)} directions, line_spacing={mil_line_spacing} mm, "
            f"sample_step={mil_sample_step} mm, valid_length={mil_valid_length} mm, "
            f"fabric_shape={getattr(args, 'mil_fabric_shape', 'ellipsoid')}, "
            f"norm_type={getattr(args, 'mil_norm_type', 'det')}, "
            f"da_mode={getattr(args, 'mil_da_mode', 'bounded')}"
        )
    else:
        print("MIL anisotropy disabled.")

    def zero_or_nan_row(center, inside_mask, use_zero=False):
        val = 0.0 if use_zero else np.nan
        return {
            "center_z_vox": center[0], "center_y_vox": center[1], "center_x_vox": center[2],
            "center_z_phys": center[0] * spacing[0], "center_y_phys": center[1] * spacing[1], "center_x_phys": center[2] * spacing[2],
            "inside_filled_mask": 1 if inside_mask else 0,
            "valid_trabfill_voxels_in_sphere": val,
            "total_voxels_in_sphere": val,
            "valid_fraction_trabfill": val,
            "distance_to_trabfill_boundary_mm": val,
            "distance_to_trabfill_boundary_vox": val,
            "BVTV": val,
            "EDTThickness_mean": val,
            "EDTThickness_max": val,
            "EDTSpacing_mean": val,
            "EDTSpacing_max": val,
            "primary_dir_st_z": val, "primary_dir_st_y": val, "primary_dir_st_x": val,
            "DA_ST_eig_1": val, "DA_ST_eig_2": val, "DA_ST_eig_3": val, "DA_ST": val,
            "angle_to_custom_long_axis_deg": val,
            "custom_long_axis_z": val, "custom_long_axis_y": val, "custom_long_axis_x": val,
            "primary_dir_st_anat_1": val, "primary_dir_st_anat_2": val, "primary_dir_st_anat_3": val,
            "primary_dir_mil_z": val, "primary_dir_mil_y": val, "primary_dir_mil_x": val,
            "DA_MIL_eig_1": val, "DA_MIL_eig_2": val, "DA_MIL_eig_3": val, "DA_MIL": val,
            "mean_intercept_length_mil": val, "mil_valid_directions": val,
            "mil_total_intersections": val, "mil_total_line_length": val,
            "angle_mil_to_custom_long_axis_deg": val,
            "primary_dir_mil_anat_1": val, "primary_dir_mil_anat_2": val, "primary_dir_mil_anat_3": val,
        }

    def analyze_one_center_local(center):
        inside_mask = center_is_valid(center, filled_mask)

        if filled_mask is not None and not inside_mask and args.center_inclusion_mode == "skip_outside":
            return None
        if filled_mask is not None and not inside_mask and args.center_inclusion_mode == "zero_outside":
            return zero_or_nan_row(center, inside_mask=False, use_zero=True)

        z, y, x = get_sphere_coords(center, sphere_offsets, bone.shape)
        if len(z) == 0:
            return zero_or_nan_row(center, inside_mask=inside_mask, use_zero=False)

        bone_vals = bone[z, y, x]
        bvtv = bvtv_in_sphere_using_filled_mask(z, y, x, bone, filled_mask=filled_mask)

        total_voxels_in_sphere = int(len(z))
        if filled_mask is not None:
            valid_trabfill_mask = filled_mask[z, y, x] > 0
        else:
            valid_trabfill_mask = np.ones_like(bone_vals, dtype=bool)
        valid_trabfill_voxels = int(np.count_nonzero(valid_trabfill_mask))
        valid_fraction_trabfill = (
            float(valid_trabfill_voxels / total_voxels_in_sphere)
            if total_voxels_in_sphere > 0 else np.nan
        )

        if filled_boundary_edt is not None:
            distance_to_boundary_mm = float(filled_boundary_edt[center])
            distance_to_boundary_vox = float(distance_to_boundary_mm / min(spacing)) if min(spacing) > 0 else np.nan
        else:
            distance_to_boundary_mm = np.nan
            distance_to_boundary_vox = np.nan

        local_bone_mask_raw = bone_vals > 0
        if getattr(args, "clip_non_bvtv_to_filled_mask", True):
            local_bone_mask = local_bone_mask_raw & valid_trabfill_mask
        else:
            local_bone_mask = local_bone_mask_raw

        tbth_mean = tbth_max = np.nan
        if np.any(local_bone_mask):
            local_thickness_vals = 2.0 * edt[z[local_bone_mask], y[local_bone_mask], x[local_bone_mask]]
            tbth_mean = float(np.mean(local_thickness_vals))
            tbth_max = float(np.max(local_thickness_vals))

        tbsp_mean = tbsp_max = np.nan
        if spacing_edt is not None:
            local_marrow_mask_raw = ~(bone_vals > 0)
            if getattr(args, "clip_non_bvtv_to_filled_mask", True):
                local_marrow_mask = local_marrow_mask_raw & valid_trabfill_mask
            else:
                local_marrow_mask = local_marrow_mask_raw
            if np.any(local_marrow_mask):
                local_spacing_vals = 2.0 * spacing_edt[z[local_marrow_mask], y[local_marrow_mask], x[local_marrow_mask]]
                tbsp_mean = float(np.mean(local_spacing_vals))
                tbsp_max = float(np.max(local_spacing_vals))

        eig_st = [np.nan, np.nan, np.nan]
        dir_st = np.array([np.nan, np.nan, np.nan], dtype=float)
        aniso_st = np.nan

        if args.use_structure_tensor and np.any(local_bone_mask):
            Jzz, Jzy, Jzx, Jyy, Jyx, Jxx = st_components
            zz, yy, xx = z[local_bone_mask], y[local_bone_mask], x[local_bone_mask]
            T_st = np.array([
                [np.mean(Jzz[zz, yy, xx]), np.mean(Jzy[zz, yy, xx]), np.mean(Jzx[zz, yy, xx])],
                [np.mean(Jzy[zz, yy, xx]), np.mean(Jyy[zz, yy, xx]), np.mean(Jyx[zz, yy, xx])],
                [np.mean(Jzx[zz, yy, xx]), np.mean(Jyx[zz, yy, xx]), np.mean(Jxx[zz, yy, xx])],
            ], dtype=np.float64)

            eig_st, vecs_st = np.linalg.eigh(T_st)
            order = np.argsort(eig_st)
            eig_st = eig_st[order]
            vecs_st = vecs_st[:, order]
            dir_st = enforce_sign_convention(vecs_st[:, 0])
            aniso_st = 1.0 - (eig_st[0] / eig_st[-1]) if eig_st[-1] > 0 else np.nan

        dir_st_anat = np.array([np.nan, np.nan, np.nan], dtype=float)
        if anat_basis is not None:
            dir_st_anat = enforce_sign_convention(transform_vector_to_anatomical(dir_st, anat_basis))

        eig_mil = np.array([np.nan, np.nan, np.nan], dtype=float)
        dir_mil = np.array([np.nan, np.nan, np.nan], dtype=float)
        aniso_mil = np.nan
        mean_mil = np.nan
        mil_valid_directions = 0
        mil_total_intersections = 0
        mil_total_line_length = 0.0
        if getattr(args, "use_mil_anisotropy", False) and np.any(local_bone_mask):
            mil_result = local_mil_fabric(
                bone=bone,
                center=center,
                radius_phys=float(args.sphere_radius),
                spacing=spacing,
                directions=mil_directions,
                line_spacing=float(mil_line_spacing),
                sample_step=float(mil_sample_step),
                min_intersections=int(getattr(args, "mil_min_intersections", 2)),
                da_mode=str(getattr(args, "mil_da_mode", "bounded")),
                method=str(getattr(args, "mil_method", "standard")),
                valid_length=float(mil_valid_length),
                fabric_shape=str(getattr(args, "mil_fabric_shape", "ellipsoid")),
                norm_type=str(getattr(args, "mil_norm_type", "det")),
            )
            eig_mil = mil_result["eig"]
            dir_mil = mil_result["dir"]
            aniso_mil = mil_result["anisotropy"]
            mean_mil = mil_result["mean_mil"]
            mil_valid_directions = mil_result["valid_directions"]
            mil_total_intersections = mil_result["total_intersections"]
            mil_total_line_length = mil_result["total_line_length"]

        dir_mil_anat = np.array([np.nan, np.nan, np.nan], dtype=float)
        if anat_basis is not None:
            dir_mil_anat = enforce_sign_convention(transform_vector_to_anatomical(dir_mil, anat_basis))

        angle_custom = angle_to_axis_degrees(dir_st, custom_long_axis)
        angle_mil_custom = angle_to_axis_degrees(dir_mil, custom_long_axis)
        if custom_long_axis is None:
            custom_axis_z = custom_axis_y = custom_axis_x = np.nan
        else:
            custom_axis_z, custom_axis_y, custom_axis_x = custom_long_axis

        return {
            "center_z_vox": center[0], "center_y_vox": center[1], "center_x_vox": center[2],
            "center_z_phys": center[0] * spacing[0], "center_y_phys": center[1] * spacing[1], "center_x_phys": center[2] * spacing[2],
            "inside_filled_mask": 1 if inside_mask else 0,
            "valid_trabfill_voxels_in_sphere": valid_trabfill_voxels,
            "total_voxels_in_sphere": total_voxels_in_sphere,
            "valid_fraction_trabfill": valid_fraction_trabfill,
            "distance_to_trabfill_boundary_mm": distance_to_boundary_mm,
            "distance_to_trabfill_boundary_vox": distance_to_boundary_vox,
            "BVTV": bvtv,
            "EDTThickness_mean": tbth_mean,
            "EDTThickness_max": tbth_max,
            "EDTSpacing_mean": tbsp_mean,
            "EDTSpacing_max": tbsp_max,
            "primary_dir_st_z": dir_st[0], "primary_dir_st_y": dir_st[1], "primary_dir_st_x": dir_st[2],
            "DA_ST_eig_1": eig_st[0], "DA_ST_eig_2": eig_st[1], "DA_ST_eig_3": eig_st[2], "DA_ST": aniso_st,
            "angle_to_custom_long_axis_deg": angle_custom,
            "custom_long_axis_z": custom_axis_z, "custom_long_axis_y": custom_axis_y, "custom_long_axis_x": custom_axis_x,
            "primary_dir_st_anat_1": dir_st_anat[0], "primary_dir_st_anat_2": dir_st_anat[1], "primary_dir_st_anat_3": dir_st_anat[2],
            "primary_dir_mil_z": dir_mil[0], "primary_dir_mil_y": dir_mil[1], "primary_dir_mil_x": dir_mil[2],
            "DA_MIL_eig_1": eig_mil[0], "DA_MIL_eig_2": eig_mil[1], "DA_MIL_eig_3": eig_mil[2], "DA_MIL": aniso_mil,
            "mean_intercept_length_mil": mean_mil, "mil_valid_directions": mil_valid_directions,
            "mil_total_intersections": mil_total_intersections, "mil_total_line_length": mil_total_line_length,
            "angle_mil_to_custom_long_axis_deg": angle_mil_custom,
            "primary_dir_mil_anat_1": dir_mil_anat[0], "primary_dir_mil_anat_2": dir_mil_anat[1], "primary_dir_mil_anat_3": dir_mil_anat[2],
        }

    results = []
    kept_centers = []
    for i, center in enumerate(centers_all, start=1):
        if i == 1 or i % args.progress_every == 0 or i == len(centers_all):
            print(f"Processing sphere {i}/{len(centers_all)}")
        row = analyze_one_center_local(center)
        if row is None:
            continue
        results.append(row)
        kept_centers.append(center)

    results_df = pd.DataFrame(results)
    print("Finished sphere analysis. Rows:", len(results_df))

    # Derived BV/TV variables. These are calculated from the valid BVTV values
    # for the current object and are written to both CSV outputs.
    if "BVTV" in results_df.columns:
        bvtv_vals = pd.to_numeric(results_df["BVTV"], errors="coerce")
        mean_bvtv = float(bvtv_vals.mean(skipna=True))
        sd_bvtv = float(bvtv_vals.std(skipna=True))
        min_bvtv = float(bvtv_vals.min(skipna=True))
        max_bvtv = float(bvtv_vals.max(skipna=True))

        results_df["rBVTV"] = bvtv_vals / mean_bvtv if np.isfinite(mean_bvtv) and mean_bvtv != 0 else np.nan
        results_df["BVTV_zscore"] = (bvtv_vals - mean_bvtv) / sd_bvtv if np.isfinite(sd_bvtv) and sd_bvtv != 0 else np.nan
        results_df["BVTV_minmax"] = (
            (bvtv_vals - min_bvtv) / (max_bvtv - min_bvtv)
            if np.isfinite(min_bvtv) and np.isfinite(max_bvtv) and max_bvtv != min_bvtv
            else np.nan
        )

    csv_path = out_dir / prefixed_name(args, "trabecular_results.csv")
    results_df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    excel_csv_path = out_dir / prefixed_name(args, "trabecular_results_excel.csv")
    results_df.to_csv(
        excel_csv_path,
        index=False,
        sep=";",
        decimal=","
    )
    print("Saved Excel-friendly CSV:", excel_csv_path)

    if anat_basis is not None:
        basis_path = out_dir / prefixed_name(args, "anatomical_basis_matrix.csv")
        np.savetxt(basis_path, anat_basis, delimiter=",")
        print("Saved anatomical basis:", basis_path)

    if auto_filled_mask_created and args.save_auto_filled_mask:
        auto_mask_path = out_dir / prefixed_name(args, "auto_filled_mask.tif")
        tifffile.imwrite(auto_mask_path, filled_mask.astype(np.uint8))
        print("Saved auto-generated filled mask:", auto_mask_path)

    if args.save_point_vtp:
        vtp_path = out_dir / prefixed_name(args, "trabecular_spheres.vtp")
        export_spheres_to_vtp(results_df, str(vtp_path), use_only_inside=True)
        print("Saved VTP:", vtp_path)

    if args.save_dense_nifti:
        dense_dir = out_dir / prefixed_name(args, f"dense_nifti_{args.dense_map_mode}")
        dense_dir.mkdir(parents=True, exist_ok=True)
        paint_mask = results_df["inside_filled_mask"].to_numpy() == 1
        kept_centers_dense = [c for c, keep in zip(kept_centers, paint_mask) if keep]
        results_df_dense = results_df.loc[paint_mask].reset_index(drop=True)
        for var in [v for v in ["BVTV", "rBVTV", "BVTV_zscore", "BVTV_minmax", "valid_fraction_trabfill", "distance_to_trabfill_boundary_mm", "distance_to_trabfill_boundary_vox", "EDTThickness_mean", "EDTThickness_max", "EDTSpacing_mean", "EDTSpacing_max", "DA_ST", "angle_to_custom_long_axis_deg", "DA_MIL", "mean_intercept_length_mil", "angle_mil_to_custom_long_axis_deg"] if v in results_df_dense.columns]:
            if args.dense_map_mode == "local_sphere":
                vol = local_sphere_heatmap(bone.shape, kept_centers_dense, results_df_dense[var].tolist(), args.sphere_radius, spacing)
            else:
                vol = center_only_heatmap(bone.shape, kept_centers_dense, results_df_dense[var].tolist())
            save_nifti(vol, spacing, str(dense_dir / prefixed_name(args, f"{var}.nii.gz")))
        print("Saved dense NIfTI maps to:", dense_dir)

    return results_df, kept_centers, bone, filled_mask, spacing


# =============================================================================
# Optional tetrahedral mesh export
# =============================================================================

def require_mesh_imports():
    global pv
    if pv is None:
        raise ImportError("pyvista is required for mesh export. Install with: pip install pyvista")
    try:
        import tetgen
        from pymeshfix import MeshFix
        import pyacvd
    except Exception as e:
        raise ImportError("Mesh export requires tetgen, pymeshfix, and pyacvd. Install with: pip install tetgen pymeshfix pyacvd") from e
    return tetgen, MeshFix, pyacvd


def pad_binary_volume(volume: np.ndarray, pad_width: int = 10) -> np.ndarray:
    return np.pad(volume, pad_width=pad_width, mode="constant", constant_values=0)


def keep_largest_component(volume: np.ndarray) -> np.ndarray:
    labels, nlab = ndimage.label(volume > 0)
    if nlab == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    sizes = ndimage.sum(volume > 0, labels=labels, index=np.arange(1, nlab + 1))
    sizes = np.asarray(sizes)
    largest_label = int(np.argmax(sizes)) + 1
    return (labels == largest_label).astype(np.uint8)


def print_volume_diagnostics(volume: np.ndarray, name: str = "volume") -> None:
    print(f"{name} shape:", volume.shape)
    print(f"{name} dtype:", volume.dtype)
    print(f"{name} foreground voxels:", int(np.sum(volume)))
    coords = np.argwhere(volume > 0)
    print(f"{name} nonzero voxel count:", len(coords))
    if len(coords) > 0:
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        print(f"{name} bounding box (z,y,x):")
        print("  min:", (zmin, ymin, xmin))
        print("  max:", (zmax, ymax, xmax))
        print("  size:", (zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1))


def binary_to_surface_mesh(volume: np.ndarray, spacing_zyx: Tuple[float, float, float], origin_xyz=(0.0, 0.0, 0.0)):
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape)[::-1]
    grid.spacing = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    grid.origin = origin_xyz
    vals_xyz = np.transpose(volume, (2, 1, 0)).astype(np.float32)
    grid.point_data["values"] = vals_xyz.ravel(order="F")
    return grid.contour(isosurfaces=[0.5], scalars="values").clean().triangulate()


def repair_surface_with_meshfix(surface, verbose=True):
    _, MeshFix, _ = require_mesh_imports()
    surf = surface.triangulate().clean()
    if verbose:
        print("Input surface before MeshFix:", surf)
        print("  manifold:", surf.is_manifold)
    mfix = MeshFix(surf)
    try:
        mfix.repair(joincomp=True, remove_smallest_components=False)
    except TypeError:
        mfix.repair()
    repaired = mfix.mesh.triangulate().clean()
    if verbose:
        print("Surface after MeshFix:", repaired)
        print("  manifold:", repaired.is_manifold)
    return repaired


def remesh_surface_with_pyacvd(surface, n_surf=15000, subdivide=0, verbose=True):
    _, _, pyacvd = require_mesh_imports()
    surf = surface.triangulate().clean()
    if verbose:
        print("Input surface before pyacvd:", surf)
    clustered = pyacvd.Clustering(surf)
    if subdivide > 0:
        clustered.subdivide(subdivide)
    clustered.cluster(n_surf)
    uniform_surf = clustered.create_mesh().triangulate().clean()
    if verbose:
        print("Surface after pyacvd remeshing:", uniform_surf)
    return uniform_surf


def tetrahedralize_surface(surface, switches="pq1.2Q", verbose=True):
    tetgen, _, _ = require_mesh_imports()
    surf = surface.triangulate().clean()
    tgen = tetgen.TetGen(surf)
    if verbose:
        print("TetGen input surface:", tgen.mesh)
    tgen.tetrahedralize(switches=switches)
    tetmesh = tgen.grid
    if verbose:
        print("TetGen output grid:", tetmesh)
    return tetmesh


def get_csv_value_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    if exclude is None:
        exclude = []
    base_exclude = {
        "center_x_vox", "center_y_vox", "center_z_vox",
        "center_x_phys", "center_y_phys", "center_z_phys",
    }
    exclude_set = base_exclude.union(set(exclude))
    return [c for c in df.columns if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])]


def nearest_interpolate(query_points, source_points, source_values):
    tree = cKDTree(source_points)
    _, idx = tree.query(query_points, k=1)
    return source_values[idx]


def idw_interpolate(query_points, source_points, source_values, k=4, power=2.0, eps=1e-12):
    tree = cKDTree(source_points)
    dists, idx = tree.query(query_points, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.maximum(dists, eps) ** power
    exact = dists <= eps
    out = np.empty(query_points.shape[0], dtype=float)
    for i in range(query_points.shape[0]):
        if np.any(exact[i]):
            out[i] = source_values[idx[i][exact[i]][0]]
        else:
            w = weights[i]
            vals = source_values[idx[i]]
            out[i] = np.sum(w * vals) / np.sum(w)
    return out


def transfer_csv_to_mesh(mesh, df: pd.DataFrame, value_columns=None, target="cells", method="idw", k=4, power=2.0):
    if value_columns is None:
        value_columns = get_csv_value_columns(df)
    source_points = df[["center_x_phys", "center_y_phys", "center_z_phys"]].to_numpy(dtype=float)
    if target == "cells":
        query_points = mesh.cell_centers().points
        data_dict = mesh.cell_data
    elif target == "points":
        query_points = mesh.points
        data_dict = mesh.point_data
    else:
        raise ValueError("target must be 'cells' or 'points'")

    for col in value_columns:
        vals = df[col].to_numpy(dtype=float)
        if method == "nearest":
            interp = nearest_interpolate(query_points, source_points, vals)
        elif method == "idw":
            interp = idw_interpolate(query_points, source_points, vals, k=k, power=power)
        else:
            raise ValueError("method must be 'nearest' or 'idw'")
        data_dict[col] = interp.astype(np.float32)
    return mesh



def _smooth_surface(surface, iterations: int, relaxation: float):
    """Smooth a PyVista surface with a conservative fallback between versions."""
    if iterations is None or int(iterations) <= 0:
        return surface
    try:
        return surface.smooth(
            n_iter=int(iterations),
            relaxation_factor=float(relaxation),
            feature_smoothing=False,
            boundary_smoothing=True,
        ).clean().triangulate()
    except TypeError:
        return surface.smooth(n_iter=int(iterations), relaxation_factor=float(relaxation)).clean().triangulate()


def _decimate_surface(surface, target_reduction: float):
    """Decimate a PyVista PolyData surface while trying to preserve topology."""
    target_reduction = float(target_reduction)
    if target_reduction <= 0:
        return surface
    target_reduction = min(target_reduction, 0.99)
    try:
        return surface.decimate_pro(target_reduction, preserve_topology=True).clean().triangulate()
    except TypeError:
        return surface.decimate_pro(target_reduction).clean().triangulate()


def save_surface_mesh_binary(surface, out_path: Path) -> None:
    """Save surface mesh, using binary output when supported."""
    try:
        surface.save(str(out_path), binary=True)
    except TypeError:
        surface.save(str(out_path))


def make_smoothed_decimated_surface_mesh(volume: np.ndarray, spacing, args, label: str, out_path: Path):
    """Create a smoothed, decimated surface mesh from a binary stack.

    The input stack is interpreted in z,y,x array order. The output surface is in
    x,y,z physical coordinates, matching the other ParaView outputs.
    """
    if pv is None:
        raise ImportError("pyvista is required for surface mesh export. Install with: pip install pyvista")

    volume = (volume > 0).astype(np.uint8)

    if getattr(args, "surface_mesh_keep_largest_component", True):
        print(f"Keeping largest connected component for {label} surface mesh...")
        volume = keep_largest_component(volume)

    pad_width = int(getattr(args, "surface_mesh_pad_width", 1))
    if pad_width > 0:
        print(f"Padding {label} surface volume by {pad_width} voxels...")
        volume = pad_binary_volume(volume, pad_width=pad_width)
        origin_xyz = (
            -pad_width * spacing[2],
            -pad_width * spacing[1],
            -pad_width * spacing[0],
        )
    else:
        origin_xyz = (0.0, 0.0, 0.0)

    print(f"Extracting {label} surface mesh...")
    base_surface = binary_to_surface_mesh(volume, spacing_zyx=spacing, origin_xyz=origin_xyz)
    base_surface = base_surface.clean().triangulate()
    print(f"{label} original surface: {base_surface.n_points} points, {base_surface.n_cells} cells")

    reductions = [float(getattr(args, "surface_mesh_target_reduction", 0.90))]
    max_mb = getattr(args, "surface_mesh_max_file_size_mb", None)
    if max_mb is not None and float(max_mb) > 0:
        for extra in [0.95, 0.97, 0.98, 0.99]:
            if extra > reductions[0] and extra not in reductions:
                reductions.append(extra)

    last_surface = None
    last_path = None
    for reduction in reductions:
        surface = base_surface.copy()
        surface = _smooth_surface(
            surface,
            iterations=int(getattr(args, "surface_mesh_smooth_iterations", 30)),
            relaxation=float(getattr(args, "surface_mesh_smooth_relaxation", 0.05)),
        )
        if getattr(args, "surface_mesh_decimate", True):
            surface = _decimate_surface(surface, reduction)
        surface = surface.clean().triangulate()

        save_surface_mesh_binary(surface, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(
            f"Saved {label} surface mesh: {out_path} "
            f"({surface.n_points} points, {surface.n_cells} cells, {size_mb:.2f} MB, reduction={reduction})"
        )
        last_surface = surface
        last_path = out_path

        if max_mb is None or float(max_mb) <= 0 or size_mb <= float(max_mb):
            break

    return last_path, last_surface


def export_surface_meshes_for_object(args, spacing, segmentations_dir: Path) -> list[str]:
    """Export smoothed/decimated surface meshes for configured segmentation stacks."""
    if not getattr(args, "make_surface_meshes", False):
        return []

    suffixes = getattr(args, "surface_mesh_suffixes", ["_FILL", "_TRABFILL"])
    if isinstance(suffixes, str):
        suffixes = [s.strip() for s in suffixes.split(",") if s.strip()]

    out_dir = Path(args.out_dir) / str(getattr(args, "surface_mesh_output_dir_name", "surface_meshes"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = str(getattr(args, "surface_mesh_file_format", "vtp")).lower().lstrip(".")
    written = []

    for suffix in suffixes:
        try:
            stack_folder = find_single_folder_ending(segmentations_dir, suffix)
            label = suffix.lstrip("_")
            out_path = out_dir / prefixed_name(args, f"{label}_surface.{fmt}")
            print(f"Creating {label} surface mesh from: {stack_folder}")
            volume = load_binary_volume(str(stack_folder))
            path, _surface = make_smoothed_decimated_surface_mesh(
                volume=volume,
                spacing=spacing,
                args=args,
                label=label,
                out_path=out_path,
            )
            written.append(str(path))

            # Optional STL export. STL stores only triangle geometry
            # and is useful for downstream mesh software, while VTP can be used in
            # Paraview
            if getattr(args, "surface_mesh_save_stl", False):
                stl_path = out_path.with_suffix(".stl")
                save_surface_mesh_binary(_surface.triangulate().clean(), stl_path)
                stl_size_mb = stl_path.stat().st_size / (1024 * 1024)
                print(
                    f"Saved {label} STL surface mesh: {stl_path} "
                    f"({_surface.n_points} points, {_surface.n_cells} cells, {stl_size_mb:.2f} MB)"
                )
                written.append(str(stl_path))

            del volume, _surface
            gc.collect()
        except Exception as e:
            msg = f"Surface mesh export failed for suffix {suffix}: {e}"
            if getattr(args, "surface_mesh_fail_on_error", True):
                raise RuntimeError(msg) from e
            print("WARNING:", msg)

    return written


def _scale_cgal_length(value, spacing, units="voxels"):
    """Convert CGAL length setting to mm.

    Trabmap stores voxel spacing in mm, so voxel-based values are converted to physical units here.
    """
    if value is None:
        return None
    value = float(value)
    if units == "voxels":
        # Current mapping assumes isotropic voxels because the runner supplies
        # one voxel_size. Use the mean for safety if spacing becomes anisotropic.
        return value * float(np.mean(spacing))
    if units == "mm":
        return value
    raise ValueError('cgal_length_units must be "voxels" or "mm".')


def _meshio_to_pyvista_tetra(meshio_mesh):
    """Convert a meshio tetrahedral mesh to a PyVista UnstructuredGrid."""
    if pv is None:
        raise ImportError("pyvista is required to convert/save CGAL meshes. Install with: pip install pyvista")

    tetra_blocks = []
    for block in meshio_mesh.cells:
        if block.type in ["tetra", "tetra4"]:
            tetra_blocks.append(np.asarray(block.data, dtype=np.int64))

    if len(tetra_blocks) == 0:
        available = [block.type for block in meshio_mesh.cells]
        raise RuntimeError(f"CGAL mesh did not contain tetrahedral cells. Cell types found: {available}")

    tets = np.vstack(tetra_blocks).astype(np.int64)
    cells = np.hstack([np.full((tets.shape[0], 1), 4, dtype=np.int64), tets]).ravel()
    celltypes = np.full(tets.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    points = np.asarray(meshio_mesh.points, dtype=float)
    return pv.UnstructuredGrid(cells, celltypes, points)


def make_cgal_mesh(args, spacing):
    """Create a tetrahedral mesh using pygalmesh/CGAL.

    This backend meshes the binary TRABFILL image domain directly rather than
    first creating a dense surface and then filling it. 
    cgal_max_element_size acts as the target/maximum internal
    tetrahedron size through CGAL's max_cell_circumradius parameter.
    """
    global pv
    if pv is None:
        raise ImportError("pyvista is required for mesh export. Install with: pip install pyvista")
    try:
        import pygalmesh  # type: ignore
    except Exception as e:
        raise ImportError(
            "CGAL-style meshing requires pygalmesh, which is an optional dependency. "
            "Use mesh_backend='tetgen' or install pygalmesh/CGAL in a compatible environment."
        ) from e

    out_dir = Path(args.out_dir)
    csv_path = out_dir / prefixed_name(args, "trabecular_results.csv")
    out_mesh_path = Path(args.mesh_out) if args.mesh_out else out_dir / prefixed_name(args, "trabecular_tetra_mesh.vtk")

    print("Loading filled volume for CGAL mesh...")
    volume = load_binary_volume(args.filled_mask if args.filled_mask else str(out_dir / prefixed_name(args, "auto_filled_mask.tif")))
    print_volume_diagnostics(volume, name="CGAL original volume")

    if args.mesh_largest_component:
        print("Keeping largest connected component...")
        volume = keep_largest_component(volume).astype(bool)
        print_volume_diagnostics(volume, name="CGAL largest-component volume")

    # Pad the volume so the object is fully separated from image boundaries.
    print(f"Padding volume by {args.mesh_pad_width} voxels...")
    volume = pad_binary_volume(volume.astype(np.uint8), pad_width=args.mesh_pad_width).astype(np.uint8)
    print_volume_diagnostics(volume, name="CGAL padded volume")

    units = getattr(args, "cgal_length_units", "voxels")
    max_cell = _scale_cgal_length(getattr(args, "cgal_max_element_size", 5.0), spacing, units)
    facet_size = _scale_cgal_length(getattr(args, "cgal_facet_size", 5.0), spacing, units)
    facet_distance = _scale_cgal_length(getattr(args, "cgal_facet_distance", 0.5), spacing, units)
    facet_angle = float(getattr(args, "cgal_facet_angle", 30.0))

    # pygalmesh expects voxel_size in x,y,z order. Trabmap spacing is z,y,x.
    voxel_size_xyz = (float(spacing[2]), float(spacing[1]), float(spacing[0]))

    print("Generating CGAL image-domain tetrahedral mesh...")
    print(f"  cgal_length_units: {units}")
    print(f"  max_cell_circumradius / max element size: {max_cell} mm")
    print(f"  max_facet_size: {facet_size} mm")
    print(f"  max_facet_distance: {facet_distance} mm")
    print(f"  min_facet_angle: {facet_angle} degrees")

    # pygalmesh API names have been stable for generate_from_array, but some
    # optional optimization flags vary by version. Try the complete call first,
    # then fall back to the core parameters if needed.
    print("pygalmesh version:", pygalmesh.__version__)
    
    try:
        meshio_mesh = pygalmesh.generate_from_array(
            volume,
            voxel_size=voxel_size_xyz,
            max_facet_distance=facet_distance,
            max_cell_circumradius=max_cell,
            verbose=bool(getattr(args, "cgal_verbose", False)),
        )
    except TypeError as e:
        raise RuntimeError(
            f"Unsupported pygalmesh version. Error was:\n{e}"
        )
    # pygalmesh follows NumPy array axis order, but Trabmap volumes are z,y,x.
    # Reorder to x,y,z so the CGAL mesh matches TetGen/VTP/STL outputs.
    meshio_mesh.points = meshio_mesh.points[:, [2, 1, 0]]

    # Correct for the padding added before meshing.
    meshio_mesh.points[:, 0] -= args.mesh_pad_width * spacing[2]
    meshio_mesh.points[:, 1] -= args.mesh_pad_width * spacing[1]
    meshio_mesh.points[:, 2] -= args.mesh_pad_width * spacing[0]

    tetmesh = _meshio_to_pyvista_tetra(meshio_mesh)

    if tetmesh.n_cells < 100:
        raise RuntimeError(
            f"CGAL tetrahedral mesh looks suspiciously small ({tetmesh.n_cells} cells). "
            "Try lowering cgal_max_element_size, checking the TRABFILL mask, or disabling largest-component filtering."
        )

    print("Loading sphere CSV...")
    df = pd.read_csv(csv_path)
    required = {"center_x_phys", "center_y_phys", "center_z_phys"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required coordinate columns: {missing}")
    print("CSV rows:", len(df))

    value_columns = None if args.mesh_value_columns is None else [x.strip() for x in args.mesh_value_columns.split(",") if x.strip()]
    print(f"Transferring CSV values to CGAL mesh ({args.mesh_target}, {args.mesh_method})...")
    tetmesh = transfer_csv_to_mesh(
        tetmesh, df, value_columns=value_columns, target=args.mesh_target,
        method=args.mesh_method, k=args.mesh_k, power=args.mesh_power
    )

    # Store backend metadata as field data where VTK supports it.
    try:
        tetmesh.field_data["mesh_backend"] = np.array(["cgal"])
        tetmesh.field_data["cgal_max_element_size_mm"] = np.array([max_cell], dtype=float)
        tetmesh.field_data["cgal_facet_size_mm"] = np.array([facet_size], dtype=float)
        tetmesh.field_data["cgal_facet_distance_mm"] = np.array([facet_distance], dtype=float)
    except Exception:
        pass

    print(f"Saving CGAL tetrahedral mesh to: {out_mesh_path}")
    tetmesh.save(str(out_mesh_path))
    print("Saved CGAL mesh:")
    print(tetmesh)
    return tetmesh


def make_tetgen_mesh(args, spacing):
    """Create tetrahedral mesh with the existing TetGen surface-first backend."""
    require_mesh_imports()
    out_dir = Path(args.out_dir)
    csv_path = out_dir / prefixed_name(args, "trabecular_results.csv")
    out_mesh_path = Path(args.mesh_out) if args.mesh_out else out_dir / prefixed_name(args, "trabecular_tetra_mesh.vtk")

    print("Loading filled volume for TetGen mesh...")
    volume = load_binary_volume(args.filled_mask if args.filled_mask else str(out_dir / prefixed_name(args, "auto_filled_mask.tif")))
    print_volume_diagnostics(volume, name="original volume")

    if args.mesh_largest_component:
        print("Keeping largest connected component...")
        volume = keep_largest_component(volume)
        print_volume_diagnostics(volume, name="largest-component volume")

    print(f"Padding volume by {args.mesh_pad_width} voxels...")
    volume = pad_binary_volume(volume, pad_width=args.mesh_pad_width)
    print_volume_diagnostics(volume, name="padded volume")

    origin_xyz = (
        -args.mesh_pad_width * spacing[2],
        -args.mesh_pad_width * spacing[1],
        -args.mesh_pad_width * spacing[0],
    )
    print("Using origin correction (x,y,z):", origin_xyz)

    print("Extracting surface mesh...")
    surface = binary_to_surface_mesh(volume, spacing_zyx=spacing, origin_xyz=origin_xyz)
    print(surface)
    print("Surface points:", surface.n_points)
    print("Surface faces:", surface.n_cells)
    print("Surface bounds:", surface.bounds)

    if args.meshfix:
        print("Repairing surface with MeshFix...")
        surface = repair_surface_with_meshfix(surface, verbose=True)

    if args.pyacvd:
        print("Uniform remeshing with pyacvd...")
        surface = remesh_surface_with_pyacvd(surface, n_surf=args.pyacvd_n_surf, subdivide=args.pyacvd_subdivide, verbose=True)

    print("Tetrahedralizing with TetGen...")
    tetmesh = tetrahedralize_surface(surface, switches=args.tetgen_switches, verbose=True)
    if tetmesh.n_cells < 100:
        raise RuntimeError(
            f"Tetrahedral mesh looks suspiciously small ({tetmesh.n_cells} cells). "
            "Try increasing mesh_pad_width, enabling meshfix, lowering pyacvd_n_surf, or changing tetgen_switches."
        )

    print("Loading sphere CSV...")
    df = pd.read_csv(csv_path)
    required = {"center_x_phys", "center_y_phys", "center_z_phys"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required coordinate columns: {missing}")
    print("CSV rows:", len(df))

    value_columns = None if args.mesh_value_columns is None else [x.strip() for x in args.mesh_value_columns.split(",") if x.strip()]
    print(f"Transferring CSV values to TetGen mesh ({args.mesh_target}, {args.mesh_method})...")
    tetmesh = transfer_csv_to_mesh(
        tetmesh, df, value_columns=value_columns, target=args.mesh_target,
        method=args.mesh_method, k=args.mesh_k, power=args.mesh_power
    )

    try:
        tetmesh.field_data["mesh_backend"] = np.array(["tetgen"])
    except Exception:
        pass

    print(f"Saving TetGen tetrahedral mesh to: {out_mesh_path}")
    tetmesh.save(str(out_mesh_path))
    print("Saved TetGen mesh:")
    print(tetmesh)
    return tetmesh


def make_tetra_mesh(args, spacing):
    """Dispatch tetrahedral mesh creation to selected backend."""
    backend = getattr(args, "mesh_backend", "tetgen")
    print(f"Mesh backend: {backend}")
    if backend == "tetgen":
        return make_tetgen_mesh(args, spacing)
    if backend == "cgal":
        return make_cgal_mesh(args, spacing)
    raise ValueError('mesh_backend must be "tetgen" or "cgal".')



# =============================================================================
# Save diagnistic images for debugging
# =============================================================================


def normalize_to_uint8_for_debug(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize an array to 8-bit for visual debug TIFFs."""
    arr = np.asarray(arr, dtype=np.float32)

    if mask is not None:
        vals = arr[np.asarray(mask, dtype=bool)]
    else:
        vals = arr[np.isfinite(arr)]

    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    vmin = float(np.percentile(vals, 1))
    vmax = float(np.percentile(vals, 99))

    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def save_debug_mid_slices(arr: np.ndarray, out_dir: Path, name: str, mask: Optional[np.ndarray] = None) -> None:
    """Save Z/Y/X mid-slices of a 3D array as 8-bit TIFFs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr8 = normalize_to_uint8_for_debug(arr, mask=mask)

    z_mid = arr8.shape[0] // 2
    y_mid = arr8.shape[1] // 2
    x_mid = arr8.shape[2] // 2

    tifffile.imwrite(out_dir / f"{name}_Zmid.tif", arr8[z_mid])
    tifffile.imwrite(out_dir / f"{name}_Ymid.tif", arr8[:, y_mid, :])
    tifffile.imwrite(out_dir / f"{name}_Xmid.tif", arr8[:, :, x_mid])



# =============================================================================
# Batch runner helpers
# =============================================================================

def detect_csv_delimiter(csv_path: Path) -> str:
    """Detect whether the runner CSV is comma- or semicolon-separated."""
    sample = csv_path.read_text(encoding="utf-8-sig")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except Exception:
        return ";" if sample.count(";") > sample.count(",") else ","


def find_single_folder_ending(parent: Path, suffix: str) -> Path:
    """Find exactly one immediate subfolder of parent ending with suffix."""
    matches = [p for p in parent.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if len(matches) == 0:
        raise FileNotFoundError(f"No folder ending with {suffix!r} found in: {parent}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise RuntimeError(f"More than one folder ending with {suffix!r} found in {parent}: {names}")
    return matches[0]


CUSTOM_ANATOMICAL_FRAME_COLUMNS = [
    "landmark_long_1_x", "landmark_long_1_y", "landmark_long_1_z",
    "landmark_long_2_x", "landmark_long_2_y", "landmark_long_2_z",
    "landmark_rotate_x", "landmark_rotate_y", "landmark_rotate_z",
]


def row_has_complete_custom_anatomical_frame(row: pd.Series) -> bool:
    """Return True when all nine custom anatomical-frame landmark values are present."""
    if not all(col in row.index for col in CUSTOM_ANATOMICAL_FRAME_COLUMNS):
        return False
    return all(str(row[col]).strip() not in ["", "nan", "None"] for col in CUSTOM_ANATOMICAL_FRAME_COLUMNS)


def get_custom_anatomical_frame_from_row(row: pd.Series):
    """Read custom anatomical-frame landmarks from runner row as x,y,z pixel coordinates.

    Returns
    -------
    tuple
        (p1_xyz, p2_xyz, prot_xyz) when all columns are present and numeric;
        otherwise (None, None, None) when columns are absent or blank.
    """
    if not row_has_complete_custom_anatomical_frame(row):
        return None, None, None

    try:
        p1_xyz = (
            float(str(row["landmark_long_1_x"]).strip()),
            float(str(row["landmark_long_1_y"]).strip()),
            float(str(row["landmark_long_1_z"]).strip()),
        )
        p2_xyz = (
            float(str(row["landmark_long_2_x"]).strip()),
            float(str(row["landmark_long_2_y"]).strip()),
            float(str(row["landmark_long_2_z"]).strip()),
        )
        prot_xyz = (
            float(str(row["landmark_rotate_x"]).strip()),
            float(str(row["landmark_rotate_y"]).strip()),
            float(str(row["landmark_rotate_z"]).strip()),
        )
    except ValueError as e:
        raise ValueError(
            "Custom anatomical-frame landmark columns are present but one or more values "
            "could not be converted to numbers."
        ) from e

    return p1_xyz, p2_xyz, prot_xyz


def copy_settings_for_object(
    base_settings,
    object_name: str,
    voxel_size: float,
    trab_folder: Path,
    trabfill_folder: Path,
    out_dir: Path,
    custom_long_axis_p1_xyz=None,
    custom_long_axis_p2_xyz=None,
    custom_rotation_point_xyz=None,
):
    """Create an object-specific settings namespace from the top-level SETTINGS."""
    d = dict(vars(base_settings))

    # Values determined from the runner CSV and folder structure.
    d["input"] = str(trab_folder)
    d["filled_mask"] = str(trabfill_folder)
    d["out_dir"] = str(out_dir)
    d["spacing_z"] = float(voxel_size)
    d["spacing_y"] = float(voxel_size)
    d["spacing_x"] = float(voxel_size)
    d["object"] = object_name
    d["output_prefix"] = f"{object_name}_"
    d["segmentations_dir"] = str(trab_folder.parent)

    if (
        custom_long_axis_p1_xyz is not None
        and custom_long_axis_p2_xyz is not None
        and custom_rotation_point_xyz is not None
    ):
        d["use_custom_anatomical_frame"] = True
        d["custom_long_axis_p1_xyz"] = custom_long_axis_p1_xyz
        d["custom_long_axis_p2_xyz"] = custom_long_axis_p2_xyz
        d["custom_rotation_point_xyz"] = custom_rotation_point_xyz
    else:
        d["use_custom_anatomical_frame"] = False
        d["custom_long_axis_p1_xyz"] = None
        d["custom_long_axis_p2_xyz"] = None
        d["custom_rotation_point_xyz"] = None

    # Do not let a mesh path from another run accidentally redirect output.
    if d.get("mesh_out") in [None, "", "None"]:
        d["mesh_out"] = None

    return SimpleNamespace(**d)

def simple_setting_value(value):
    """Convert settings to CSV-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return ",".join(str(x) for x in value)
    return value



def flatten_config_values(config, prefix=""):
    """Flatten a config dictionary/list into key-value pairs for runner CSV logging."""
    if config is None:
        return {}
    if isinstance(config, SimpleNamespace):
        config = vars(config)
    flat = {}
    if isinstance(config, dict):
        for key, value in config.items():
            safe_key = str(key).strip().replace(" ", "_").replace("-", "_")
            new_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
            flat.update(flatten_config_values(value, new_prefix))
    elif isinstance(config, (list, tuple)):
        flat[prefix] = simple_setting_value(config)
    else:
        flat[prefix] = simple_setting_value(config)
    return flat


def write_config_values_to_runner_row(df: pd.DataFrame, row_index, args) -> None:
    """Add/update every user-supplied config value in the runner dataframe."""
    config_values = getattr(args, "config_values", None)
    if not config_values:
        return
    for key, value in flatten_config_values(config_values).items():
        col = f"config_{key}"
        if col not in df.columns:
            df[col] = ""
        df.at[row_index, col] = simple_setting_value(value)


def write_settings_to_runner_row(df: pd.DataFrame, row_index, args) -> None:
    """Add/update setting columns in the runner dataframe for the current object."""
    exclude = {"runner_csv", "input_root", "output_heatmap_root"}
    for key, value in vars(args).items():
        if key in exclude:
            continue
        col = f"setting_{key}"
        if col not in df.columns:
            df[col] = ""
        df.at[row_index, col] = simple_setting_value(value)


def save_runner(df: pd.DataFrame, runner_csv: Path, delimiter: str) -> None:
    """Save runner CSV back to disk."""
    df.to_csv(runner_csv, index=False, sep=delimiter)


def run_from_runner(base_settings):
    """Read trabmap_runner.csv and process every row where finished is not yes."""
    runner_csv = Path(base_settings.runner_csv)

    # Stop immediately if Excel or another program has locked the runner CSV.
    # This avoids completing analyses but then failing to write finished/status columns.
    check_runner_csv_writable(runner_csv)


    # Outputs are saved inside each object folder:
    # INPUT_ROOT / object_name / mappings


    input_root = Path(base_settings.input_root)

    if not runner_csv.exists():
        raise FileNotFoundError(f"Runner CSV not found: {runner_csv}")

    if not input_root.exists():
        raise FileNotFoundError(f"Input root folder not found: {input_root}")

    delimiter = detect_csv_delimiter(runner_csv)
    runner = pd.read_csv(runner_csv, sep=delimiter, dtype=str).fillna("")

    required_cols = {"finished", "object", "voxel_size"}
    missing = required_cols - set(runner.columns)
    if missing:
        raise ValueError(f"Runner CSV is missing required columns: {missing}")

    for col in ["status", "error_message", "input_folder", "filled_mask_folder", "output_folder", "custom_anatomical_frame_used", "custom_long_axis_used", "surface_meshes_status", "surface_meshes_files"]:
        if col not in runner.columns:
            runner[col] = ""

    to_run = runner[runner["finished"].str.strip().str.lower() != "yes"]
    print(f"Runner rows total: {len(runner)}")
    print(f"Rows still to process: {len(to_run)}")

    for row_index, row in to_run.iterrows():
        object_name = str(row["object"]).strip()
        if object_name == "":
            runner.at[row_index, "status"] = "skipped"
            runner.at[row_index, "error_message"] = "Empty object field"
            save_runner(runner, runner_csv, delimiter)
            continue

        print("\n" + "=" * 80)
        print(f"Starting object: {object_name}")
        print("=" * 80)

        try:
            voxel_size = float(str(row["voxel_size"]).strip())
            object_dir = input_root / object_name
            if not object_dir.exists():
                raise FileNotFoundError(f"Object folder not found: {object_dir}")

            segmentations_dir = object_dir / "segmentations"

            if not segmentations_dir.exists():
                raise FileNotFoundError(f"Segmentations folder not found: {segmentations_dir}")

            trab_folder = find_single_folder_ending(segmentations_dir, "_TRAB")
            trabfill_folder = find_single_folder_ending(segmentations_dir, "_TRABFILL")

            out_dir = object_dir / "mappings"
            out_dir.mkdir(parents=True, exist_ok=True)

            custom_p1_xyz, custom_p2_xyz, custom_rot_xyz = get_custom_anatomical_frame_from_row(row)

            args = copy_settings_for_object(
                base_settings=base_settings,
                object_name=object_name,
                voxel_size=voxel_size,
                trab_folder=trab_folder,
                trabfill_folder=trabfill_folder,
                out_dir=out_dir,
                custom_long_axis_p1_xyz=custom_p1_xyz,
                custom_long_axis_p2_xyz=custom_p2_xyz,
                custom_rotation_point_xyz=custom_rot_xyz,
            )
            validate_settings(args)
            if args.use_custom_anatomical_frame:
                print(
                    "Using custom anatomical frame from runner CSV: "
                    f"long axis {args.custom_long_axis_p1_xyz} -> {args.custom_long_axis_p2_xyz}; "
                    f"rotation point {args.custom_rotation_point_xyz}"
                )
            else:
                print("No complete custom anatomical-frame landmarks for this object; anatomical-frame columns will not be calculated.")

            runner.at[row_index, "input_folder"] = str(trab_folder)
            runner.at[row_index, "filled_mask_folder"] = str(trabfill_folder)
            runner.at[row_index, "output_folder"] = str(out_dir)
            runner.at[row_index, "custom_anatomical_frame_used"] = "yes" if args.use_custom_anatomical_frame else "no"
            runner.at[row_index, "custom_long_axis_used"] = "yes" if args.use_custom_anatomical_frame else "no"
            runner.at[row_index, "status"] = "running"
            runner.at[row_index, "error_message"] = ""
            write_settings_to_runner_row(runner, row_index, args)
            write_config_values_to_runner_row(runner, row_index, args)
            save_runner(runner, runner_csv, delimiter)

            results_df, kept_centers, bone, filled_mask, spacing = analyze_spheres(args)

            surface_mesh_files = []
            if getattr(args, "make_surface_meshes", False):
                surface_mesh_files = export_surface_meshes_for_object(
                    args=args,
                    spacing=spacing,
                    segmentations_dir=Path(args.segmentations_dir),
                )
                runner.at[row_index, "surface_meshes_status"] = "finished"
                runner.at[row_index, "surface_meshes_files"] = " | ".join(surface_mesh_files)
                write_settings_to_runner_row(runner, row_index, args)
                write_config_values_to_runner_row(runner, row_index, args)
                save_runner(runner, runner_csv, delimiter)

            if args.make_mesh:
                make_tetra_mesh(args, spacing)

            runner.at[row_index, "finished"] = "yes"
            runner.at[row_index, "status"] = "finished"
            runner.at[row_index, "error_message"] = ""
            write_settings_to_runner_row(runner, row_index, args)
            write_config_values_to_runner_row(runner, row_index, args)
            save_runner(runner, runner_csv, delimiter)
            print(f"Finished object: {object_name}")

        except Exception as e:
            runner.at[row_index, "status"] = "failed"
            runner.at[row_index, "error_message"] = str(e)
            save_runner(runner, runner_csv, delimiter)
            print(f"FAILED object: {object_name}")
            print(str(e))
            # Continue with the next row rather than stopping the full batch.
            continue

    print("\nBatch runner finished.")

# =============================================================================
# Run script
# =============================================================================

def main():
    validate_settings(SETTINGS)
    run_from_runner(SETTINGS)


if __name__ == "__main__":
    main()



# =============================================================================
# CONFIG-FRIENDLY ENTRY POINT
# =============================================================================

def run_from_config(config):
    """Run batch mapping after applying a config dictionary to SETTINGS."""
    from .config import namespace_from_defaults
    args = namespace_from_defaults(SETTINGS, config)
    # Preserve the raw user-supplied YAML/JSON values so every configurable
    # input can be written back to the runner CSV for reproducibility.
    args.config_values = dict(config)
    validate_settings(args)
    return run_from_runner(args)
