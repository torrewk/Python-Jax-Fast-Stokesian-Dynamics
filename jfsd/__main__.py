#!/usr/bin/env python

import argparse
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

from jfsd.config import JfsdConfiguration
from jfsd.interactive import interactive_main
from jfsd.main import main

DEFAULT_CONFIG_FP = files(__package__) / "default_configuration.toml"


def cli_main():
    parser = argparse.ArgumentParser(
        prog="jfsd", description="Run particle simulations use fast JIT compilation."
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Set the configuration file of your simulation.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--start-configuration",
        help="Provide a starting configuration of the particles.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="Set the parameters interactively with questions.",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--output", help="Output directory to store the results in.", type=Path, default=None
    )
    args = parser.parse_args()
    if args.interactive:
        interactive_main()
        return
    if args.config is None:
        config_fp = DEFAULT_CONFIG_FP
    else:
        config_fp = args.config

    config = JfsdConfiguration.from_toml(config_fp, args.start_configuration)

    main(**config.parameters, output=args.output)


if __name__ == "__main__":
    cli_main()


def _parse_config(config):
    final_dict = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            final_dict.update(_parse_config(value))
        else:
            final_dict[key] = value
    return final_dict
