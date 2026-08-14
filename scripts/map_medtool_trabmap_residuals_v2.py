#!/usr/bin/env python
r"""
Map Medtool–TrabMap regression residuals onto a 3D mesh.

Inputs are the combined VTK and CSV created by compare_medtool_trabmap_mapping.py.
For each variable pair, the script fits TrabMap ~ Medtool, then writes signed
residuals, absolute residuals, raw differences, zero/missing flags, and fitted
values back onto the mesh and into a node-level CSV.

to run 
python scripts\map_medtool_trabmap_residuals_v2.py --combined-vtk "C:\Users\Jaaps\Documents\Trabmap\trabmap_github\mt_tm_comparison\results\medtool_trabmap_comparison.vtk" --combined-csv "C:\Users\Jaaps\Documents\Trabmap\trabmap_github\mt_tm_comparison\results\medtool_trabmap_comparison.csv"

"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv

VARIABLE_PAIRS = {
    # Medtool provides both BVTV_voxel and BVTV_smooth. BVTV_smooth is placed
    # first because it is the closer analogue to TrabMap's sphere-wise mapped
    # BV/TV values. Reverse the order if you specifically want BVTV_voxel.
    "BVTV": {
        "medtool": ["BVTV_smooth", "BVTV_voxel", "BVTV", "BV/TV"],
        "trabmap": ["BVTV_trabmap"],
    },

    # TrabMap DA_MIL uses the bounded definition 1 - lambda_min/lambda_max.
    # Medtool's DA_val01 is therefore the appropriate comparison field.
    "DA_MIL": {
        "medtool": ["DA_val01", "DA_value", "DA", "DA_MIL"],
        "trabmap": ["DA_mil_trabmap", "DA_MIL_trabmap"],
    },

    # This is an exploratory cross-method comparison: TrabMap DA_ST versus
    # Medtool's bounded MIL-derived DA. It is not a like-for-like algorithmic
    # comparison, but it is useful for mapping where the two anisotropy methods
    # diverge spatially.
    "DA_ST": {
        "medtool": ["DA_val01", "DA_value", "DA", "DA_MIL"],
        "trabmap": ["DA_st_trabmap", "DA_ST_trabmap"],
    },

    "TbTh": {
        "medtool": ["Tb_Th_mean", "TbTh", "Tb.Th", "Thickness"],
        "trabmap": ["TbTh_trabmap", "EDTThickness_mean_trabmap"],
    },

    "TbSp": {
        "medtool": ["Tb_Sp_mean", "TbSp", "Tb.Sp", "Spacing"],
        "trabmap": ["TbSp_trabmap", "EDTSpacing_mean_trabmap"],
    },
}

def parse_args():
    p=argparse.ArgumentParser(description="Fit Medtool–TrabMap regressions and map residuals to a VTK mesh.")
    p.add_argument("--combined-vtk", required=True, type=Path)
    p.add_argument("--combined-csv", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--include-zero-pairs", action="store_true")
    p.add_argument("--standardized-residuals", action="store_true")
    return p.parse_args()

def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def resolve(columns, aliases):
    exact=set(columns)
    for a in aliases:
        if a in exact: return a
    lookup={norm(c): c for c in columns}
    for a in aliases:
        if norm(a) in lookup: return lookup[norm(a)]
    return None

def identify_pairs(df):
    out={}
    for label,c in VARIABLE_PAIRS.items():
        m=resolve(df.columns, c["medtool"]); t=resolve(df.columns, c["trabmap"])
        if m is None or t is None:
            print(f"WARNING: skipping {label}; Medtool={m}, TrabMap={t}")
            continue
        out[label]=(m,t); print(f"Resolved {label}: {m} vs {t}")
    if not out: raise KeyError(f"No comparison pairs found. Columns: {list(df.columns)}")
    return out

def check_alignment(mesh, df):
    if mesh.n_points != len(df):
        raise ValueError(f"VTK has {mesh.n_points} points but CSV has {len(df)} rows.")
    if all(c in df.columns for c in ("x","y","z")):
        xyz=df[["x","y","z"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid=np.isfinite(xyz).all(1)
        if np.any(valid):
            d=np.linalg.norm(xyz[valid]-np.asarray(mesh.points)[valid], axis=1)
            md=float(np.max(d)); print(f"Maximum CSV-to-VTK coordinate difference: {md:.6g}")
            if md>1e-5: raise ValueError(f"CSV and VTK coordinates do not align; max difference={md:g}")

def fit_and_map(df, med_col, trab_col, label, include_zero, standardized):
    x=pd.to_numeric(df[med_col], errors="coerce").to_numpy(float)
    y=pd.to_numeric(df[trab_col], errors="coerce").to_numpy(float)
    finite=np.isfinite(x)&np.isfinite(y)
    exclude_zero=np.zeros(len(df), bool)
    if not include_zero and label!="BVTV":
        exclude_zero=finite&((x==0)|(y==0))
    mask=finite&~exclude_zero
    if mask.sum()<3: raise ValueError(f"Fewer than 3 valid pairs for {label}")
    X=np.column_stack([np.ones(mask.sum()), x[mask]])
    b,*_=np.linalg.lstsq(X, y[mask], rcond=None)
    intercept,slope=map(float,b)
    pred=np.full(len(df), np.nan); resid=np.full(len(df), np.nan)
    pred[mask]=intercept+slope*x[mask]; resid[mask]=y[mask]-pred[mask]
    absres=np.abs(resid)
    raw=np.full(len(df), np.nan); raw[finite]=y[finite]-x[finite]
    absraw=np.abs(raw)
    rvalid=resid[mask]
    ss_res=float(np.sum(rvalid**2)); ss_tot=float(np.sum((y[mask]-np.mean(y[mask]))**2))
    r2=float(1-ss_res/ss_tot) if ss_tot>0 else np.nan
    rsd=float(np.std(rvalid, ddof=2)) if mask.sum()>2 else np.nan
    arrays={
        f"{label}_medtool":x,
        f"{label}_trabmap":y,
        f"{label}_predicted_trabmap":pred,
        f"{label}_regression_residual":resid,
        f"{label}_absolute_regression_residual":absres,
        f"{label}_raw_difference":raw,
        f"{label}_absolute_raw_difference":absraw,
        f"{label}_included_in_regression":mask.astype(np.uint8),
        f"{label}_excluded_zero_pair":exclude_zero.astype(np.uint8),
        f"{label}_medtool_zero":(np.isfinite(x)&(x==0)).astype(np.uint8),
        f"{label}_trabmap_zero":(np.isfinite(y)&(y==0)).astype(np.uint8),
        f"{label}_medtool_missing":(~np.isfinite(x)).astype(np.uint8),
        f"{label}_trabmap_missing":(~np.isfinite(y)).astype(np.uint8),
    }
    if standardized:
        z=np.full(len(df), np.nan)
        if np.isfinite(rsd) and rsd>0: z[mask]=resid[mask]/rsd
        arrays[f"{label}_standardized_residual"]=z
    summary={
        "variable":label,"medtool_column":med_col,"trabmap_column":trab_col,
        "n_total_nodes":len(df),"n_finite_pairs":int(finite.sum()),"n_regression":int(mask.sum()),
        "n_excluded_zero_pairs":int(exclude_zero.sum()),
        "n_medtool_zero":int((np.isfinite(x)&(x==0)).sum()),"n_trabmap_zero":int((np.isfinite(y)&(y==0)).sum()),
        "n_medtool_missing":int((~np.isfinite(x)).sum()),"n_trabmap_missing":int((~np.isfinite(y)).sum()),
        "intercept":intercept,"slope":slope,"r_squared":r2,
        "rmse_regression":float(np.sqrt(np.mean(rvalid**2))),
        "mae_regression":float(np.mean(np.abs(rvalid))),
        "residual_sd":rsd,"mean_signed_residual":float(np.mean(rvalid)),
        "maximum_absolute_residual":float(np.max(np.abs(rvalid))),
    }
    return arrays,summary

def main():
    a=parse_args(); vtk=a.combined_vtk.resolve(); csv=a.combined_csv.resolve()
    if not vtk.exists(): raise FileNotFoundError(vtk)
    if not csv.exists(): raise FileNotFoundError(csv)
    out=(a.output_dir.resolve() if a.output_dir else vtk.parent/"residual_mapping"); out.mkdir(parents=True, exist_ok=True)
    mesh=pv.read(vtk); df = pd.read_csv(csv,sep=None,engine="python"); check_alignment(mesh, df); pairs=identify_pairs(df)
    result_mesh=mesh.copy(); result_df=df.copy(); summaries=[]
    for label,(m,t) in pairs.items():
        print(f"\nFitting {label}...")
        arrays,summary=fit_and_map(df,m,t,label,a.include_zero_pairs,a.standardized_residuals)
        for n,v in arrays.items(): result_mesh.point_data[n]=v; result_df[n]=v
        summaries.append(summary)
        print(f"  n={summary['n_regression']}, intercept={summary['intercept']:.6g}, slope={summary['slope']:.6g}, R2={summary['r_squared']:.4f}")
    vtk_out=out/"medtool_trabmap_regression_residuals.vtk"
    csv_out=out/"medtool_trabmap_regression_residuals.csv"
    sum_out=out/"medtool_trabmap_regression_summary.csv"
    result_mesh.save(vtk_out); result_df.to_csv(csv_out,index=False); pd.DataFrame(summaries).to_csv(sum_out,index=False)
    print(f"\nSaved:\n{vtk_out}\n{csv_out}\n{sum_out}")
    print("\nInspect *_absolute_regression_residual, *_regression_residual, *_trabmap_zero, and *_excluded_zero_pair in ParaView.")
    return 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr); raise
