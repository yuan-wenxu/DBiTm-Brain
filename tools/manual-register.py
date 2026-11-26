#!/usr/bin/env python3
"""Build a self-contained registration HTML from three full-resolution images."""

import argparse
from pathlib import Path

from manual_registration import build_registration, registration_parser


def parse_args() -> argparse.Namespace:
    parser = registration_parser(__doc__)
    parser.add_argument("--taps-image", type=Path, required=True)
    parser.add_argument("--taps-beta-image", type=Path, required=True)
    args = parser.parse_args()
    if args.max_image_edge < 256:
        parser.error("--max-image-edge must be at least 256")
    return args


def main() -> None:
    args = parse_args()
    image_specs = (
        (args.mrna_image, "mrna", "mRNA"),
        (args.taps_image, "taps", "TAPS"),
        (args.taps_beta_image, "taps_beta", "TAPS-beta"),
    )
    build_registration(args, image_specs, "taps_beta", "manual-registration.html")


if __name__ == "__main__":
    main()
