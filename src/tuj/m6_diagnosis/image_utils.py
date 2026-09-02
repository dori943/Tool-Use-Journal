"""Image helpers for OpenAI VLM failure diagnosis."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
_EXTENSION_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class ImagePathError(ValueError):
    """Raised when an observation image reference cannot be resolved."""


def _mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_type = _EXTENSION_TO_MIME.get(suffix)
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/png", "image/jpeg"}:
        raise ImagePathError(
            f"unsupported image type for {path}; supported extensions: "
            f"{sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    return mime_type


def local_image_to_data_url(path: str | Path) -> str:
    """Read a local image file and return a base64 data URL."""
    image_path = Path(path)
    if not image_path.exists():
        raise ImagePathError(f"observation image not found: {image_path}")
    if not image_path.is_file():
        raise ImagePathError(f"observation image path is not a file: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImagePathError(
            f"unsupported image extension {suffix!r} for {image_path}; "
            f"supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    mime_type = _mime_type_for_path(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_observation_image(path: str | Path | None) -> str | None:
    """Resolve an observation image reference to a data URL, or return None."""
    if path is None:
        return None
    if isinstance(path, str) and not path.strip():
        return None
    return local_image_to_data_url(path)


def resolve_observation_images(observation: dict) -> list[tuple[str, str]]:
    """Resolve before/after observation images for OpenAI vision input."""
    resolved: list[tuple[str, str]] = []
    before_url = resolve_observation_image(observation.get("before_image"))
    if before_url is not None:
        resolved.append(("before_image", before_url))
    after_url = resolve_observation_image(observation.get("after_image"))
    if after_url is not None:
        resolved.append(("after_image", after_url))
    return resolved
