# Trabmap

Trabmap is an open-source Python pipeline for semi-automated cortical/trabecular bone segmentation and whole-bone trabecular structure mapping. The segmentation algorithm is a modified version of the algorithm of Buie et al. (2007). The method of quantification of trabecular bone properties is modified from the algorithm outlined by Gross et al. (2014).

References: 
Buie, H. R., Campbell, G. M., Klinck, R. J., MacNeil, J. A., & Boyd, S. K. (2007). Automatic segmentation of cortical and trabecular compartments based on a dual threshold technique for in vivo micro-CT bone analysis. Bone, 41(4), 505-515.

Gross, T., Kivell, T. L., Skinner, M. M., Nguyen, H., & Pahr, D. H. (2014). A CT-image-based framework for the holistic analysis of cortical and trabecular bone morphology. Palaeo electronica 17.3.33A


## Repository layout

below is a layout of the folder structures, important files, and what they do:
```text
trabmap/
├── config/trabmap_config.yaml     # edit paths and parameters here
├── data/trabmap_runner.csv        # runner file, add specimen name and voxel size here
├── scripts/run_segmentation.py    # script wrapper
├── scripts/run_mapping.py         # script wrapper
└── src/trabmap/
    ├── segmentation.py            # segmentation functions
    ├── mapping.py                 # sphere mapping, VTP, VTK mesh export functions
    ├── config.py                  # YAML/JSON config helpers
    ├── cli_segment.py             # command-line interface
    └── cli_map.py                 # command-line interface
```

## Installation
download the folder structure to your local computre, open the terminal, and navigate to the folder where trabmap lives using cd and pasting your folder location. Then, in the terminal, run the following commands to install a trabmap environment with all the required libraries
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
    ├── OBJECT_NAME_TRABCORT/      # initial bone stacke
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

Edit `config/trabmap_config.yaml` first, especially the following parameters:

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
python scripts/run_segmentation.py --config config/trabmap_config.yaml
python scripts/run_mapping.py --config config/trabmap_config.yaml
```
