from __future__ import annotations
import argparse
from .config import load_config
from . import segmentation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run trabmap cortical/trabecular segmentation.")
    parser.add_argument("--config", default="config/trabmap_config.yaml", help="Path to YAML/JSON config file.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, section="segmentation")
    segmentation.run_from_config(cfg)


if __name__ == "__main__":
    main()
