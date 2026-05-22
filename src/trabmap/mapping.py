"""
TrabMap local script: BV/TV, trabecular thickness, structure tensor direction,
ParaView VTP export, and optional tetrahedral VTK mesh export.

The script reads trabmap_runner.csv, finds rows where finished is not equal to "yes",
auto-detects the matching _TRAB and _TRABFILL folders, runs Trabmap, prefixes
outputs with the object name, and writes the settings/status back to the runner CSV.

Notes:
- Input and filled-mask can be a folder of .tif/.tiff slices, a 3D .tif/.tiff stack, or a .npy file.
- Coordinates are assumed to be z,y,x in image arrays, and x,y,z in ParaView outputs.
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
# SETTINGS: EDIT THIS SECTION FOR YOUR WORKSTATION
# =============================================================================

SETTINGS = SimpleNamespace(
    # -------------------------------------------------------------------------
    # Batch runner paths
    # -------------------------------------------------------------------------
    # CSV file with at least these columns: finished, object, voxel_size.
    # The uploaded example uses semicolons, which this script detects automatically.
    runner_csv="trabmap_runner.csv",

    # Folder containing a subfolder per object that ends with _TRABMAP 
    input_root="input",

    # Output parent folder. Each object gets its own subfolder here. old code
    output_heatmap_root=None,

    # -------------------------------------------------------------------------
    # Optional anatomical landmarks
    # -------------------------------------------------------------------------
    # Keep as None unless every object should use the same landmarks CSV.
    # If you need object-specific landmarks later, we can add a landmarks_csv
    # column to the runner file.
    # no longer used, to remove
    landmarks_csv=None,

    # -------------------------------------------------------------------------
    # Optional custom anatomical axis settings
    # -------------------------------------------------------------------------
    # These are read from trabmap_runner.csv columns:
    # landmark_long_1_x, landmark_long_1_y, landmark_long_1_z,
    # landmark_long_2_x, landmark_long_2_y, landmark_long_2_z,
    # landmark_rotate_x, landmark_rotate_y, landmark_rotate_z.
    #
    # The two long-axis landmarks define anatomical axis 1.
    # this can be used to determine the orientation of the trabecular structure
    # relative to a specified long axis defined by two points and a rotation axis
    # defined bz another point
    # example: long axis can define proximodistal axis while the rotation axis
    # is sued to specify where anterior is
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
    sphere_radius=5.0, # default is 5.0
    grid_step=2.5, # default is 2.5
    margin=None,

    # What to do with grid centers outside the filled mask:
    # "skip_outside" = ignore them; "zero_outside" = keep them with zero values.
    # skip outside is a sensible default, zero outside is to make output
    # comparable to ORS Dragonfly
    center_inclusion_mode="skip_outside",

    # -------------------------------------------------------------------------
    # Structure tensor settings
    # -------------------------------------------------------------------------
    use_structure_tensor=True,
    st_sigma=1.0,

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
    # Example: "BVTV,TbTh_mean_local,anisotropy_st"
    mesh_value_columns=None,

    mesh_pad_width=20,
    mesh_largest_component=True,
    meshfix=True,
    pyacvd=True,
    pyacvd_n_surf=7500, # default was 15000, more likely to fail so set to 7500, this produces a slightly less detail surface mesh
    pyacvd_subdivide=0,
    tetgen_switches="pq1.2Q", # original value, likely results in many vertices on outside fewer inside but smaller file size
    #tetgen_switches="pq1.2a0.05Q" # .05Q constrains inside vertex size, making them more equal to the surface of the mesh, reduce to make intenral even smaller
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


def prefixed_name(args, filename: str) -> str:
    """Return filename with args.output_prefix added, if present."""
    return f"{getattr(args, 'output_prefix', '')}{filename}"


# =============================================================================
# General helpers
# =============================================================================

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

    Returns
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

    for col in [c for c in ["BVTV", "TbTh_mean_local", "TbTh_p90_local", "TbTh_p95_local", "TbTh_max_local", "anisotropy_st", "angle_to_custom_long_axis_deg"] if c in df.columns]:
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

    if "anisotropy_st" in df.columns and "principal_direction" in cloud.array_names:
        cloud["dir_weighted"] = cloud["principal_direction"] * df["anisotropy_st"].to_numpy()[:, None]

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

    print("Computing EDT...")
    edt = ndimage.distance_transform_edt(bone, sampling=spacing)
    print("EDT min/max:", float(np.min(edt)), float(np.max(edt)))

    st_components = None
    if args.use_structure_tensor:
        print("Computing structure tensor...")
        st_image = edt.astype(np.float32) * bone.astype(np.float32)
        st_components = structure_tensor_components_3d(st_image, sigma=args.st_sigma)
        del st_image
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

    def zero_or_nan_row(center, inside_mask, use_zero=False):
        val = 0.0 if use_zero else np.nan
        return {
            "center_z_vox": center[0], "center_y_vox": center[1], "center_x_vox": center[2],
            "center_z_phys": center[0] * spacing[0], "center_y_phys": center[1] * spacing[1], "center_x_phys": center[2] * spacing[2],
            "inside_filled_mask": 1 if inside_mask else 0,
            "BVTV": val,
            "TbTh_mean_local": val,
            "TbTh_p90_local": val,
            "TbTh_p95_local": val,
            "TbTh_max_local": val,
            "primary_dir_st_z": val, "primary_dir_st_y": val, "primary_dir_st_x": val,
            "eig_st_1": val, "eig_st_2": val, "eig_st_3": val, "anisotropy_st": val,
            "angle_to_custom_long_axis_deg": val,
            "custom_long_axis_z": val, "custom_long_axis_y": val, "custom_long_axis_x": val,
            "primary_dir_st_anat_1": val, "primary_dir_st_anat_2": val, "primary_dir_st_anat_3": val,
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
        local_bone_mask = bone_vals > 0

        tbth_mean = tbth_p90 = tbth_p95 = tbth_max = np.nan
        if np.any(local_bone_mask):
            local_thickness_vals = 2.0 * edt[z[local_bone_mask], y[local_bone_mask], x[local_bone_mask]]
            tbth_mean = float(np.mean(local_thickness_vals))
            tbth_p90 = float(np.percentile(local_thickness_vals, 90))
            tbth_p95 = float(np.percentile(local_thickness_vals, 95))
            tbth_max = float(np.max(local_thickness_vals))

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

        angle_custom = angle_to_axis_degrees(dir_st, custom_long_axis)
        if custom_long_axis is None:
            custom_axis_z = custom_axis_y = custom_axis_x = np.nan
        else:
            custom_axis_z, custom_axis_y, custom_axis_x = custom_long_axis

        return {
            "center_z_vox": center[0], "center_y_vox": center[1], "center_x_vox": center[2],
            "center_z_phys": center[0] * spacing[0], "center_y_phys": center[1] * spacing[1], "center_x_phys": center[2] * spacing[2],
            "inside_filled_mask": 1 if inside_mask else 0,
            "BVTV": bvtv,
            "TbTh_mean_local": tbth_mean,
            "TbTh_p90_local": tbth_p90,
            "TbTh_p95_local": tbth_p95,
            "TbTh_max_local": tbth_max,
            "primary_dir_st_z": dir_st[0], "primary_dir_st_y": dir_st[1], "primary_dir_st_x": dir_st[2],
            "eig_st_1": eig_st[0], "eig_st_2": eig_st[1], "eig_st_3": eig_st[2], "anisotropy_st": aniso_st,
            "angle_to_custom_long_axis_deg": angle_custom,
            "custom_long_axis_z": custom_axis_z, "custom_long_axis_y": custom_axis_y, "custom_long_axis_x": custom_axis_x,
            "primary_dir_st_anat_1": dir_st_anat[0], "primary_dir_st_anat_2": dir_st_anat[1], "primary_dir_st_anat_3": dir_st_anat[2],
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
        for var in [v for v in ["BVTV", "TbTh_mean_local", "TbTh_p90_local", "TbTh_p95_local", "TbTh_max_local", "anisotropy_st", "angle_to_custom_long_axis_deg"] if v in results_df_dense.columns]:
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


def make_tetra_mesh(args, spacing):
    require_mesh_imports()
    out_dir = Path(args.out_dir)
    csv_path = out_dir / prefixed_name(args, "trabecular_results.csv")
    out_mesh_path = Path(args.mesh_out) if args.mesh_out else out_dir / prefixed_name(args, "trabecular_tetra_mesh.vtk")

    print("Loading filled volume for mesh...")
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

    print("Tetrahedralizing...")
    tetmesh = tetrahedralize_surface(surface, switches=args.tetgen_switches, verbose=True)
    if tetmesh.n_cells < 100:
        raise RuntimeError(
            f"Tetrahedral mesh looks suspiciously small ({tetmesh.n_cells} cells). "
            "Try increasing --mesh-pad-width, enabling --meshfix, lowering --pyacvd-n-surf, or changing --tetgen-switches."
        )

    print("Loading sphere CSV...")
    df = pd.read_csv(csv_path)
    required = {"center_x_phys", "center_y_phys", "center_z_phys"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required coordinate columns: {missing}")
    print("CSV rows:", len(df))

    value_columns = None if args.mesh_value_columns is None else [x.strip() for x in args.mesh_value_columns.split(",") if x.strip()]
    print(f"Transferring CSV values to mesh ({args.mesh_target}, {args.mesh_method})...")
    tetmesh = transfer_csv_to_mesh(
        tetmesh, df, value_columns=value_columns, target=args.mesh_target,
        method=args.mesh_method, k=args.mesh_k, power=args.mesh_power
    )

    print(f"Saving tetrahedral mesh to: {out_mesh_path}")
    tetmesh.save(str(out_mesh_path))
    print("Saved mesh:")
    print(tetmesh)



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

    for col in ["status", "error_message", "input_folder", "filled_mask_folder", "output_folder", "custom_anatomical_frame_used", "custom_long_axis_used"]:
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
            save_runner(runner, runner_csv, delimiter)

            results_df, kept_centers, bone, filled_mask, spacing = analyze_spheres(args)
            if args.make_mesh:
                make_tetra_mesh(args, spacing)

            runner.at[row_index, "finished"] = "yes"
            runner.at[row_index, "status"] = "finished"
            runner.at[row_index, "error_message"] = ""
            write_settings_to_runner_row(runner, row_index, args)
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
    validate_settings(args)
    return run_from_runner(args)
