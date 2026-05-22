# TrabMap

Trabmap is an open-source Python pipeline for semi-automated cortical/trabecular bone segmentation and whole-bone trabecular structure mapping.

This repository version reorganizes the original working scripts into a GitHub package.

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
 Here you can also alter variables necessary to get a high quality segmentation. These are set to reasonable defaults but may need to be altered to suit a specific bone or anatomical region

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

## What changed from the original scripts?

- Hard-coded workstation paths were removed.
- Parameters now live in `config/trabmap_config.yaml`.
- The two working scripts were converted into importable modules.
- Command-line entry points were added.
- The existing runner CSV logic is preserved.

## Notes

The mapping module has optional dependencies for mesh export: `pyvista`, `tetgen`, `pymeshfix`, and `pyacvd`. If you only need CSV/VTP outputs, you can set `make_mesh: false` in the config.
