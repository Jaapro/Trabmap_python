# Trabmap

Trabmap is an open-source Python pipeline for semi-automated cortical/trabecular bone segmentation and whole-bone trabecular structure mapping. The segmentation algorithm is a modified version of the algorithm of Buie et al. (2007). The method of quantification of trabecular bone properties is modified from the algorithm outlined by Gross et al. (2014).

References: 
Buie, H. R., Campbell, G. M., Klinck, R. J., MacNeil, J. A., & Boyd, S. K. (2007). Automatic segmentation of cortical and trabecular compartments based on a dual threshold technique for in vivo micro-CT bone analysis. Bone, 41(4), 505-515.

Gross, T., Kivell, T. L., Skinner, M. M., Nguyen, H., & Pahr, D. H. (2014). A CT-image-based framework for the holistic analysis of cortical and trabecular bone morphology. Palaeo electronica 17.3.33A https://doi.org/10.26879/438


## Repository layout

below is a layout of the folder structures, important files, and what they are used for.
```text
trabmap/
├── config/trabmap_config.yaml     # edit paths and parameters here
├── data/trabmap_runner.csv        # example runner CSV
├── scripts/run_segmentation.py    # script wrapper
├── scripts/run_mapping.py         # script wrapper
└── src/trabmap/
    ├── segmentation.py            # segmentation functions and batch logic
    ├── mapping.py                 # sphere mapping, VTP, VTK mesh export
    ├── config.py                  # YAML/JSON config helpers
    ├── cli_segment.py             # command-line interface
    └── cli_map.py                 # command-line interface
```

## Installation
download the folder structure to your local computre, open the terminal, and navigate to the folder where trabmap
was downloaded using cd and pasting your folder location.
then inthe terminal, run the following commands to install a trabmap environment with all the required libraries
```bash
conda env create -f environment.yml
conda activate trabmap
```

Or install in editable mode in an existing environment:

```bash
pip install -e .
```

## Input folder structure

The default config expects one folder per object under `input/`:

```text
input/
└── OBJECT_NAME/
    ├── OBJECT_NAME_TRABCORT/      # initial binary bone stack for segmentation
    └── segmentations/             # created by segmentation step
```

OBJECT_NAME should be entered in the trabmap_runner.csv file in the object column. The runner file is found in trabmap/data. You should also set its voxel size in the voxel_size column, in mm

The segmentation step writes:

```text
input/OBJECT_NAME/segmentations/
├── OBJECT_NAME_FILL/
├── OBJECT_NAME_CORT/
├── OBJECT_NAME_TRAB/
├── OBJECT_NAME_TRABFILL/
├── OBJECT_NAME_MARROW/
└── OBJECT_NAME_DEBUG/
```

The mapping step reads folders ending with `*_TRAB` and `*_TRABFILL` from the segmentations folder and writes mapping outputs to:

```text
input/OBJECT_NAME/mappings/
```

When `make_surface_meshes: true`, the mapping step also writes smoothed and
decimated surface meshes for `_FILL` and `_TRABFILL` to:

```text
input/OBJECT_NAME/mappings/surface_meshes/
```

By default each surface is saved as both `.vtp` and `.stl`. VTP is preferred
for ParaView/PyVista workflows, while STL is useful for downstream mesh
software that expects triangle geometry only.

## Basic use

Edit `config/trabmap_config.yaml` first, especially:

```yaml
segmentation:
  runner_csv: data/trabmap_runner.csv
  input_root: input

mapping:
  runner_csv: data/trabmap_runner.csv
  input_root: input
```
 Here you can also alter variables necessary to get a high quality segmentation. These are set to reasonable defaults but may need to be altered to suit a specific bone size or anatomical region.

To begin the segmentation and mapping process, run:

```bash
trabmap-segment --config config/trabmap_config.yaml
trabmap-map --config config/trabmap_config.yaml
```

Equivalent script-wrapper commands:

```bash
python scripts/run_segmentation.py --config config/trabmap_config.yaml
python scripts/run_mapping.py --config config/trabmap_config.yaml
```

## What changed from previous versions:

- Parameters are now stored and can be altered in `config/trabmap_config.yaml`.
- Multiple working scripts were converted into importable modules.


## Surface mesh outputs

The mapping step can also export smoothed and decimated surface meshes from the segmentation stacks. By default, it creates surfaces for:

```text
input/OBJECT_NAME/segmentations/OBJECT_NAME_FILL/
input/OBJECT_NAME/segmentations/OBJECT_NAME_TRABFILL/
```

and writes them to:

```text
input/OBJECT_NAME/mappings/surface_meshes/
├── OBJECT_NAME_FILL_surface.vtp
└── OBJECT_NAME_TRABFILL_surface.vtp
```

These outputs are controlled in `config/trabmap_config.yaml` under the `mapping:` section:

```yaml
make_surface_meshes: true
surface_mesh_suffixes: [_FILL, _TRABFILL]
surface_mesh_smooth_iterations: 30
surface_mesh_smooth_relaxation: 0.05
surface_mesh_decimate: true
surface_mesh_target_reduction: 0.90
surface_mesh_max_file_size_mb: 5.0
```

Increase `surface_mesh_target_reduction` for smaller files, or decrease it for more detailed surfaces.

## Reproducibility logging

During mapping, TrabMap writes two types of provenance columns back to the runner CSV:

- `setting_*` columns record the object-specific settings actually used for that row.
- `config_*` columns record every user-supplied value from the `mapping:` section of the YAML/JSON config file.

This makes the runner CSV an analysis log that records both the input data paths and all configurable mapping parameters used for each object.

## Notes

The mapping module has optional dependencies for mesh export: `pyvista`, `tetgen`, `pymeshfix`, and `pyacvd`. If you only need CSV/VTP outputs, you can set `make_mesh: false` in the config.

The following are non-standard trabecular properties that are useful for comparing interspecific variation in bone volume fraction across whole bones:
- rBVTV = BVTV / mean(BVTV)
- BVTV_zscore = (BVTV - mean(BVTV)) / sd(BVTV)
- BVTV_minmax = (BVTV - min(BVTV)) / (max(BVTV) - min(BVTV))