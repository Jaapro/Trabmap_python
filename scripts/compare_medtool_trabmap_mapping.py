#!/usr/bin/env python
r"""
Merge Medtool VTK files and interpolate TrabMap CSV measurements onto Medtool nodes.

Outputs
-------
<root>/results/medtool_trabmap_comparison.vtk
<root>/results/medtool_trabmap_comparison.csv

Requirements
------------
pip install numpy pandas scipy pyvista vtk

Example
-------
conda activate trabmap
cd 
python scripts\compare_medtool_trabmap_mapping.py --root "C:\Users\Jaaps\Documents\Trabmap\trabmap_github\mt_tm_comparison" --trabmap-csv "C:\Users\Jaaps\Documents\Trabmap\trabmap_github\mt_tm_comparison\TxState_D01_2011_Tibia_Dist_L_trabecular_results.csv"
Notes
-----
1. Medtool VTK coordinates and TrabMap CSV coordinates must use the same physical units,
   axis order, origin, and orientation.
2. TrabMap values are interpolated to Medtool mesh points (nodes).
3. Existing Medtool cell data are converted to point data before export.
4. Inverse-distance weighting is the default; nearest-neighbour is optional.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

VARIABLE_ALIASES = {
    "BVTV_trabmap": ["BVTV", "BV/TV", "bvtv"],
    "TbTh_trabmap": ["EDTThickness_mean", "Tb.Th", "TbTh", "trabecular_thickness"],
    "TbSp_trabmap": ["EDTSpacing_mean", "Tb.Sp", "TbSp", "trabecular_spacing"],
    "DA_mil_trabmap": ["DA_MIL", "DAmil", "DA_mil", "DA MIL"],
    "DA_st_trabmap": ["DA_ST", "DAst", "DA_st", "DA ST"],
}

COORDINATE_SETS = [
    ("center_x_phys", "center_y_phys", "center_z_phys"),
    ("x", "y", "z"),
    ("X", "Y", "Z"),
    ("coord_x", "coord_y", "coord_z"),
    ("point_x", "point_y", "point_z"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Medtool VTK files and interpolate TrabMap measurements onto Medtool nodes."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--trabmap-csv", required=True, type=Path)
    parser.add_argument("--method", choices=("idw", "nearest"), default="idw")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--max-distance", type=float, default=None)
    parser.add_argument("--csv-separator", default=None)
    parser.add_argument("--vtk-pattern", default="*.vtk")
    parser.add_argument(
        "--medtool-coordinate-tolerance",
        type=float,
        default=1e-6,
        help="Maximum coordinate difference for matching nodes among Medtool VTK files. Default: 1e-6.",
    )
    return parser.parse_args()


def find_vtk_files(root: Path, pattern: str, results_dir: Path) -> list[Path]:
    files = [p for p in root.rglob(pattern) if p.is_file() and results_dir not in p.parents]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No VTK files matching {pattern!r} were found below: {root}")
    return files


def safe_array_name(existing_names: set[str], requested_name: str, source_stem: str) -> str:
    """Return a unique point-data name without silently overwriting another array."""
    name = str(requested_name).strip() or source_stem
    if name not in existing_names:
        return name

    candidate = f"{name}_{source_stem}"
    counter = 2
    while candidate in existing_names:
        candidate = f"{name}_{source_stem}_{counter}"
        counter += 1
    return candidate


def convert_all_data_to_points(mesh: pv.DataSet) -> pv.DataSet:
    """
    Preserve original point data and additionally convert all cell arrays to point arrays.

    Cell arrays that share a name with point arrays receive a '_from_cells'
    suffix before conversion.
    """
    mesh = mesh.copy()
    point_names = set(str(n) for n in mesh.point_data.keys())

    for name in list(mesh.cell_data.keys()):
        original_name = str(name)
        if original_name in point_names:
            new_name = f"{original_name}_from_cells"
            counter = 2
            while new_name in point_names or new_name in mesh.cell_data:
                new_name = f"{original_name}_from_cells_{counter}"
                counter += 1
            mesh.cell_data[new_name] = np.asarray(mesh.cell_data[original_name])
            del mesh.cell_data[original_name]

    if len(mesh.cell_data) > 0:
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)

    return mesh


def map_source_array_to_reference(
    source_points: np.ndarray,
    source_values: np.ndarray,
    reference_points: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transfer a point array from a source Medtool mesh to the reference node set
    using nearest coordinate matching.
    """
    source_values = np.asarray(source_values)
    if source_values.shape[0] != source_points.shape[0]:
        raise ValueError(
            f"Array length {source_values.shape[0]} does not match "
            f"source point count {source_points.shape[0]}."
        )

    tree = cKDTree(source_points)
    distances, indices = tree.query(reference_points, k=1)

    if np.any(distances > tolerance):
        n_bad = int(np.count_nonzero(distances > tolerance))
        max_dist = float(np.max(distances))
        raise ValueError(
            f"{n_bad:,} reference nodes could not be matched within tolerance "
            f"{tolerance:g}; maximum nearest-node distance was {max_dist:g}. "
            "The Medtool VTK files may not share the same geometry or coordinate system."
        )

    return source_values[indices], distances


def combine_medtool_files(
    vtk_files: Iterable[Path],
    coordinate_tolerance: float = 1e-6,
) -> pv.DataSet:
    """
    Combine all original Medtool variables onto one reference mesh.

    The first VTK file supplies the geometry and topology. Every point-data array
    from every other VTK is transferred to that reference node set by coordinate
    matching. Cell data are first converted to point data.
    """
    vtk_files = list(vtk_files)
    if not vtk_files:
        raise RuntimeError("No Medtool VTK files were supplied.")

    print(f"Using reference Medtool geometry: {vtk_files[0]}")
    reference = convert_all_data_to_points(pv.read(vtk_files[0]))

    if reference.n_points == 0:
        raise RuntimeError(f"Reference Medtool mesh has zero points: {vtk_files[0]}")

    reference = reference.copy()
    reference_points = np.asarray(reference.points, dtype=float)
    existing_names: set[str] = set()

    # Preserve arrays from the reference file, but rebuild them explicitly so
    # naming collisions can be handled consistently.
    reference_arrays = {
        str(name): np.asarray(reference.point_data[name]).copy()
        for name in reference.point_data.keys()
    }
    reference.point_data.clear()

    for name, values in reference_arrays.items():
        output_name = safe_array_name(existing_names, name, vtk_files[0].stem)
        reference.point_data[output_name] = values
        existing_names.add(output_name)
        print(f"  Medtool array: {vtk_files[0].name} :: {name} -> {output_name}")

    # Add arrays from all remaining files to the same reference nodes.
    for vtk_path in vtk_files[1:]:
        print(f"Reading additional Medtool VTK: {vtk_path}")
        source = convert_all_data_to_points(pv.read(vtk_path))
        if source.n_points == 0:
            print(f"  WARNING: skipping mesh with zero points: {vtk_path}")
            continue

        source_points = np.asarray(source.points, dtype=float)
        last_distances = None

        for original_name in source.point_data.keys():
            values = np.asarray(source.point_data[original_name])
            mapped, distances = map_source_array_to_reference(
                source_points=source_points,
                source_values=values,
                reference_points=reference_points,
                tolerance=coordinate_tolerance,
            )
            last_distances = distances

            output_name = safe_array_name(
                existing_names,
                str(original_name),
                vtk_path.stem,
            )
            reference.point_data[output_name] = mapped
            existing_names.add(output_name)
            print(f"  Medtool array: {vtk_path.name} :: {original_name} -> {output_name}")

        if last_distances is not None:
            diagnostic_name = safe_array_name(
                existing_names,
                f"medtool_node_match_distance_{vtk_path.stem}",
                vtk_path.stem,
            )
            reference.point_data[diagnostic_name] = last_distances
            existing_names.add(diagnostic_name)

    reference.point_data["medtool_reference_node_id"] = np.arange(
        reference.n_points, dtype=np.int64
    )

    print(
        f"Combined Medtool mesh: {reference.n_points:,} points, "
        f"{reference.n_cells:,} cells, "
        f"{len(reference.point_data):,} point-data arrays"
    )
    print("Combined Medtool point arrays:")
    for name in reference.point_data.keys():
        print(f"  {name}")

    return reference

def read_trabmap_csv(path: Path, separator: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"TrabMap CSV not found: {path}")
    df = pd.read_csv(path, sep=separator, engine="python") if separator is not None else pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError(f"TrabMap CSV contains no rows: {path}")
    return df


def detect_coordinate_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    for cols in COORDINATE_SETS:
        if all(c in df.columns for c in cols):
            return cols
    raise KeyError("Could not identify TrabMap coordinate columns. Available columns: " + str(df.columns.tolist()))


def resolve_variable_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    lower_lookup = {str(c).lower(): str(c) for c in df.columns}
    for output_name, aliases in VARIABLE_ALIASES.items():
        source_name = None
        for alias in aliases:
            if alias in df.columns:
                source_name = alias
                break
            source_name = lower_lookup.get(alias.lower())
            if source_name is not None:
                break
        if source_name is None:
            print(f"WARNING: no TrabMap column found for {output_name}. Tried aliases: {aliases}")
            continue
        resolved[output_name] = source_name
    if not resolved:
        raise KeyError("None of the requested TrabMap variables were found.")
    return resolved


def prepare_trabmap_data(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x_col, y_col, z_col = detect_coordinate_columns(df)
    resolved = resolve_variable_columns(df)
    coords_df = df[[x_col, y_col, z_col]].apply(pd.to_numeric, errors="coerce")
    valid_coords = np.isfinite(coords_df.to_numpy(dtype=float)).all(axis=1)
    if "inside_filled_mask" in df.columns:
        inside = pd.to_numeric(df["inside_filled_mask"], errors="coerce").fillna(0).to_numpy() == 1
        valid_coords &= inside
    coords = coords_df.to_numpy(dtype=float)[valid_coords]
    if len(coords) == 0:
        raise ValueError("No valid TrabMap coordinate rows remain after filtering.")
    values: dict[str, np.ndarray] = {}
    for output_name, source_name in resolved.items():
        values[output_name] = pd.to_numeric(df[source_name], errors="coerce").to_numpy(dtype=float)[valid_coords]
    print(f"TrabMap samples retained: {len(coords):,}; coordinates: {x_col}, {y_col}, {z_col}")
    for output_name, source_name in resolved.items():
        print(f"  {source_name} -> {output_name} ({np.isfinite(values[output_name]).sum():,} finite values)")
    return coords, values


def interpolate_nearest(source_coords, source_values, target_coords, max_distance):
    valid = np.isfinite(source_values)
    if not np.any(valid):
        n = len(target_coords)
        return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=np.int32)
    tree = cKDTree(source_coords[valid])
    distances, indices = tree.query(target_coords, k=1)
    output = source_values[valid][indices].astype(float)
    if max_distance is not None:
        output[distances > max_distance] = np.nan
    return output, distances.astype(float), np.isfinite(output).astype(np.int32)


def interpolate_idw(source_coords, source_values, target_coords, k, power, max_distance):
    valid = np.isfinite(source_values)
    if not np.any(valid):
        n = len(target_coords)
        return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=np.int32)
    coords = source_coords[valid]
    vals = source_values[valid]
    k_use = min(max(1, int(k)), len(coords))
    tree = cKDTree(coords)
    distances, indices = tree.query(target_coords, k=k_use)
    if k_use == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    nearest_distance = distances[:, 0].astype(float)
    output = np.full(len(target_coords), np.nan, dtype=float)
    neighbor_count = np.zeros(len(target_coords), dtype=np.int32)
    exact_rows = distances[:, 0] == 0
    if np.any(exact_rows):
        output[exact_rows] = vals[indices[exact_rows, 0]]
        neighbor_count[exact_rows] = 1
    nonexact_rows = ~exact_rows
    if np.any(nonexact_rows):
        d = distances[nonexact_rows].astype(float)
        idx = indices[nonexact_rows]
        neighbor_vals = vals[idx]
        finite = np.isfinite(d) & np.isfinite(neighbor_vals)
        weights = np.zeros_like(d, dtype=float)
        weights[finite] = 1.0 / np.power(np.maximum(d[finite], np.finfo(float).eps), power)
        weight_sums = weights.sum(axis=1)
        usable = weight_sums > 0
        interpolated = np.full(d.shape[0], np.nan, dtype=float)
        interpolated[usable] = np.sum(weights[usable] * neighbor_vals[usable], axis=1) / weight_sums[usable]
        output[nonexact_rows] = interpolated
        neighbor_count[nonexact_rows] = finite.sum(axis=1).astype(np.int32)
    if max_distance is not None:
        too_far = nearest_distance > max_distance
        output[too_far] = np.nan
        neighbor_count[too_far] = 0
    return output, nearest_distance, neighbor_count


def add_interpolated_trabmap_data(mesh, trabmap_coords, trabmap_values, method, k, power, max_distance):
    mesh = mesh.copy()
    targets = np.asarray(mesh.points, dtype=float)
    all_distances = []
    all_counts = []
    for output_name, source_values in trabmap_values.items():
        print(f"Interpolating {output_name} to {len(targets):,} Medtool nodes...")
        if method == "nearest":
            result, distance, count = interpolate_nearest(trabmap_coords, source_values, targets, max_distance)
        else:
            result, distance, count = interpolate_idw(trabmap_coords, source_values, targets, k, power, max_distance)
        mesh.point_data[output_name] = result
        all_distances.append(distance)
        all_counts.append(count)
        print(f"  finite interpolated values: {np.isfinite(result).sum():,}/{len(result):,}")
    mesh.point_data["trabmap_nearest_sample_distance"] = np.nanmin(np.vstack(all_distances), axis=0)
    mesh.point_data["trabmap_interpolation_neighbor_count"] = np.max(np.vstack(all_counts), axis=0).astype(np.int32)
    return mesh


def array_to_csv_columns(data, base_name, n_rows):
    arr = np.asarray(data)
    if arr.shape[0] != n_rows:
        return {}
    if arr.ndim == 1:
        return {base_name: arr}
    if arr.ndim == 2:
        return {f"{base_name}_{i + 1}": arr[:, i] for i in range(arr.shape[1])}
    print(f"WARNING: skipping point-data array {base_name!r} with shape {arr.shape}")
    return {}


def export_node_csv(mesh, out_path: Path) -> None:
    points = np.asarray(mesh.points, dtype=float)
    table = {"node_id": np.arange(mesh.n_points, dtype=np.int64), "x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
    for name in mesh.point_data.keys():
        for col_name, values in array_to_csv_columns(np.asarray(mesh.point_data[name]), str(name), mesh.n_points).items():
            table[col_name] = values
    df = pd.DataFrame(table)
    df.to_csv(out_path, index=False)
    print(f"Saved node table: {out_path} ({len(df):,} rows)")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    csv_path = args.trabmap_csv.resolve()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    vtk_out = results_dir / "medtool_trabmap_comparison.vtk"
    csv_out = results_dir / "medtool_trabmap_comparison.csv"
    vtk_files = find_vtk_files(root, args.vtk_pattern, results_dir)
    print(f"Found {len(vtk_files)} Medtool VTK file(s).")
    merged = combine_medtool_files(
        vtk_files,
        coordinate_tolerance=args.medtool_coordinate_tolerance,
    )
    trabmap_df = read_trabmap_csv(csv_path, args.csv_separator)
    trabmap_coords, trabmap_values = prepare_trabmap_data(trabmap_df)
    combined = add_interpolated_trabmap_data(merged, trabmap_coords, trabmap_values, args.method, args.k, args.power, args.max_distance)
    combined.save(vtk_out)
    print(f"Saved combined VTK: {vtk_out}")
    export_node_csv(combined, csv_out)
    print("\nCompleted successfully.")
    print(f"VTK: {vtk_out}")
    print(f"CSV: {csv_out}")
    print("\nInspect 'trabmap_nearest_sample_distance' and the overlaid distributions before running regressions.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
