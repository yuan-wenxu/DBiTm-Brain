#!/usr/bin/env python3
"""Build a self-contained manual ssDNA registration interface."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
from PIL import Image


MAX_IMAGE_EDGE_PX = 1600
HTML_TEMPLATE = (
    Path(__file__).parent.parent
    / "registration-html"
    / "manual-registration.html"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrna", type=Path, required=True)
    parser.add_argument("--taps", type=Path, required=True)
    parser.add_argument("--taps-beta", dest="taps_beta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return args


def read_ssdna_image(
    path: Path,
    modality_id: str,
    label: str,
) -> dict[str, object]:
    print(f"Reading {label}: {path}", flush=True)
    adata = ad.read_h5ad(path, backed="r")
    try:
        library = next(iter(adata.uns["spatial"].values()))
        result = {
            "id": modality_id,
            "label": label,
            "tissueImage": Image.fromarray(
                np.asarray(library["images"]["hires"])
            ).convert("L"),
        }
    finally:
        adata.file.close()
    print(
        f"  hires image {result['tissueImage'].width:,} x "
        f"{result['tissueImage'].height:,}",
        flush=True,
    )
    return result


def encode_ssdna_images(
    layers: list[dict[str, object]],
) -> list[dict[str, object]]:
    payloads = []
    for layer in layers:
        source = layer["tissueImage"]
        resize_scale = min(
            1.0,
            MAX_IMAGE_EDGE_PX / max(source.width, source.height),
        )
        width = max(1, round(source.width * resize_scale))
        height = max(1, round(source.height * resize_scale))
        image = source.resize((width, height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        payloads.append(
            {
                "id": layer["id"],
                "label": layer["label"],
                "image": "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii"),
                "hiresWidth": source.width,
                "hiresHeight": source.height,
            }
        )
        print(
            f"  Embedded {layer['label']} display image {width:,} x {height:,}",
            flush=True,
        )
    return payloads


def write_registration_html(
    payloads: list[dict[str, object]],
    output_path: Path,
) -> None:
    serialized = json.dumps(
        payloads,
        ensure_ascii=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = HTML_TEMPLATE.read_text(encoding="utf-8").replace(
        "window.REGISTRATION_DATA = null;",
        "window.REGISTRATION_DATA = " + serialized + ";",
        1,
    )
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"Error: output directory already exists: {output_dir}")

    try:
        layers = [
            read_ssdna_image(args.mrna.expanduser().resolve(), "mrna", "mRNA"),
            read_ssdna_image(args.taps.expanduser().resolve(), "taps", "TAPS"),
            read_ssdna_image(
                args.taps_beta.expanduser().resolve(),
                "taps_beta",
                "TAPS-beta",
            ),
        ]
        output_dir.mkdir(parents=True)
        payloads = encode_ssdna_images(layers)
        write_registration_html(payloads, output_dir / "manual-registration.html")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Wrote {output_dir / 'manual-registration.html'}")


if __name__ == "__main__":
    main()
