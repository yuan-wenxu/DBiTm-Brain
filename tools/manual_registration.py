"""Shared image preparation for the manual registration commands."""

from __future__ import annotations

import argparse
import base64
import io
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from PIL import Image


DEFAULT_MAX_IMAGE_EDGE = 1600
HTML_TEMPLATE = (
    Path(__file__).parent.parent
    / "registration-html"
    / "manual-registration.html"
)


def registration_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mrna-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-image-edge",
        type=int,
        default=DEFAULT_MAX_IMAGE_EDGE,
        help="Maximum embedded-image width or height in pixels (default: 1600).",
    )
    return parser


def encode_image(
    path: Path,
    modality_id: str,
    label: str,
    max_image_edge: int,
    *,
    allow_color: bool = False,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")

    print(f"Reading {label}: {path}", flush=True)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as source:
        allowed_modes = ("L", "RGB", "RGBA") if allow_color else ("L",)
        if source.mode not in allowed_modes:
            expected = "an 8-bit grayscale image"
            if allow_color:
                expected = "an 8-bit grayscale, RGB, or RGBA image"
            raise ValueError(
                f"Expected {expected} at {path}, found {source.mode}"
            )
        fullres_width, fullres_height = source.size
        resize_scale = min(
            1.0,
            max_image_edge / max(fullres_width, fullres_height),
        )
        display_size = (
            max(1, round(fullres_width * resize_scale)),
            max(1, round(fullres_height * resize_scale)),
        )
        display = source.resize(
            display_size,
            resample=Image.Resampling.LANCZOS,
            reducing_gap=3.0,
        )

    buffer = io.BytesIO()
    display.save(buffer, format="PNG", optimize=True)
    print(
        f"  full resolution {fullres_width:,} x {fullres_height:,}; "
        f"embedded {display.width:,} x {display.height:,}",
        flush=True,
    )
    return {
        "id": modality_id,
        "label": label,
        "image": "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "fullresWidth": fullres_width,
        "fullresHeight": fullres_height,
    }


def write_registration_html(
    payloads: list[dict[str, object]],
    reference_id: str,
    output_path: Path,
) -> None:
    serialized = json.dumps(
        {"layers": payloads, "referenceId": reference_id},
        ensure_ascii=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    sentinel = "window.REGISTRATION_DATA = null;"
    template = HTML_TEMPLATE.read_text(encoding="utf-8")
    if template.count(sentinel) != 1:
        raise ValueError(
            "Registration HTML template has an invalid data sentinel"
        )
    html = template.replace(
        sentinel,
        "window.REGISTRATION_DATA = " + serialized + ";",
        1,
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(html.encode("utf-8"))
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_registration(
    args: argparse.Namespace,
    image_specs: Sequence[tuple[Path, str, str]],
    reference_id: str,
    filename: str,
    *,
    allow_color: bool = False,
) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(
            f"Error: output directory already exists: {output_dir}"
        )

    try:
        payloads = [
            encode_image(
                path.expanduser().resolve(),
                modality_id,
                label,
                args.max_image_edge,
                allow_color=allow_color,
            )
            for path, modality_id, label in image_specs
        ]
        output_dir.mkdir(parents=True)
        output_path = output_dir / filename
        write_registration_html(payloads, reference_id, output_path)
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Wrote {output_path}")
