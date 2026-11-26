#!/usr/bin/env python3
"""Build an offline registration page for mRNA and methylation images."""

import argparse
from pathlib import Path

from manual_registration import build_registration, registration_parser


def parse_args() -> argparse.Namespace:
    parser = registration_parser(__doc__)
    parser.add_argument(
        "--methylation-image",
        type=Path,
        required=True,
        help="Fixed reference image. The mRNA image moves in this space.",
    )
    args = parser.parse_args()
    if args.max_image_edge < 256:
        parser.error("--max-image-edge must be at least 256")
    return args


def main() -> None:
    args = parse_args()
    image_specs = (
        (args.mrna_image, "mrna", "mRNA"),
        (args.methylation_image, "methylation", "Methylation"),
    )
    build_registration(
        args,
        image_specs,
        "methylation",
        "manual-registration-v2.html",
        allow_color=True,
    )


if __name__ == "__main__":
    main()
