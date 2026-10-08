from __future__ import annotations

import argparse
import logging

from . import pipeline
from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="abuse-detect")
    parser.add_argument("command", choices=["prepare", "run", "all"])
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    if args.command in ("prepare", "all"):
        pipeline.prepare(cfg)
    if args.command in ("run", "all"):
        pipeline.run(cfg)


if __name__ == "__main__":
    main()
