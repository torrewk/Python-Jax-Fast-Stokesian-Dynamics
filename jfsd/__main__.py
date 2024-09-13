#!/usr/bin/env python

import argparse
import sys
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

import tomllib

from jfsd.interactive import interactive_main
from jfsd.main import main

DEFAULT_CONFIG_FP = files(__package__) / "default_configuration.toml"


def cli_main():
    parser = argparse.ArgumentParser(
        prog="jfsd", description="Run particle simulations use fast JIT compilation."
    )
    parser.add_argument(
        "-c", "--config",
        help="Set the configuration file of your simulation.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-i", "--interactive",
        help="Set the parameters interactively with questions.",
        action="store_true",
    )
    args = parser.parse_args()
    if args.interactive:
        interactive_main()
        return
    if args.config is None:
        config_fp = DEFAULT_CONFIG_FP
    else:
        config_fp = args.config

    with open(config_fp, "rb") as handle:
        config = tomllib.load(handle)

    main(**_parse_config(config))


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
