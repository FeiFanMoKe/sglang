from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

import msgspec

logger = logging.getLogger(__name__)

_MAX_CONFIG_BYTES = 4096
MAX_WATERMARK_CONTEXT_WINDOW = 64
MAX_WATERMARKED_CONTEXTS_PER_REQUEST = 4096
MAX_SYNTHID_DEPTH = 64
WATERMARK_ALGORITHMS = ("gumbel", "synthid")


class WatermarkConfigError(ValueError):
    pass


class WatermarkServerConfig(msgspec.Struct, frozen=True, kw_only=True):
    # None marks a field the file leaves unset, so server defaults still apply.
    key: str | None = None
    key_b: str | None = None
    context_window: int | None = None
    mixing_probability: float | None = None
    max_probability: float | None = None
    default_enabled: bool | None = None
    enforce_all: bool | None = None
    algorithm: str | None = None
    synthid_depth: int | None = None
    synthid_candidate_cap: int | None = None
    synthid_compile: bool | None = None

    def __repr__(self) -> str:
        return (
            "WatermarkServerConfig(key=<redacted>, key_b=<redacted>, "
            f"context_window={self.context_window!r}, "
            f"mixing_probability={self.mixing_probability!r}, "
            f"max_probability={self.max_probability!r}, "
            f"default_enabled={self.default_enabled!r}, "
            f"enforce_all={self.enforce_all!r}, "
            f"algorithm={self.algorithm!r}, "
            f"synthid_depth={self.synthid_depth!r}, "
            f"synthid_candidate_cap={self.synthid_candidate_cap!r}, "
            f"synthid_compile={self.synthid_compile!r})"
        )


def parse_watermark_key(value: Any) -> int:
    if not isinstance(value, str):
        raise ValueError("watermark key must be a hex string")
    digits = value[2:] if value.lower().startswith("0x") else value
    if not 1 <= len(digits) <= 16:
        raise ValueError("watermark key must contain 1 to 16 hex digits")
    if any(character not in "0123456789abcdefABCDEF" for character in digits):
        raise ValueError("watermark key must contain only hex digits")
    key = int(digits, 16)
    return key if key < (1 << 63) else key - (1 << 64)


def _read_config_file(path: str) -> str:
    config_path = Path(path).expanduser()
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
        descriptor = os.open(config_path, flags)
        with os.fdopen(descriptor, encoding="utf-8") as config_file:
            file_stat = os.fstat(config_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise WatermarkConfigError(
                    "watermark config must be a readable regular file"
                )
            if stat.S_IMODE(file_stat.st_mode) & 0o077:
                logger.warning(
                    "watermark config is readable by group or other users; "
                    "restrict it to the server account"
                )
            return config_file.read(_MAX_CONFIG_BYTES + 1)
    except (OSError, UnicodeError):
        raise WatermarkConfigError("failed to read watermark config JSON") from None


def load_watermark_config(source: str) -> WatermarkServerConfig:
    if source.lstrip().startswith("{"):
        payload = source
    else:
        payload = _read_config_file(source)
    if len(payload.encode("utf-8")) > _MAX_CONFIG_BYTES:
        raise WatermarkConfigError("watermark config exceeds 4096 bytes")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        raise WatermarkConfigError("failed to read watermark config JSON") from None
    if not isinstance(raw, dict):
        raise WatermarkConfigError("watermark config must be a JSON object")
    if set(raw) - set(WatermarkServerConfig.__struct_fields__):
        raise WatermarkConfigError("watermark config contains unknown fields")

    try:
        for name in ("key", "key_b"):
            if raw.get(name) is not None:
                parse_watermark_key(raw[name])
    except ValueError as error:
        raise WatermarkConfigError(str(error)) from error
    context_window = raw.get("context_window")
    if context_window is not None and (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or not 1 <= context_window <= MAX_WATERMARK_CONTEXT_WINDOW
    ):
        raise WatermarkConfigError(
            "watermark config context_window must be an integer from 1 to 64"
        )
    for name in ("mixing_probability", "max_probability"):
        value = raw.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise WatermarkConfigError(f"watermark config {name} must be a number")
    for name in ("default_enabled", "enforce_all", "synthid_compile"):
        if raw.get(name) is not None and not isinstance(raw[name], bool):
            raise WatermarkConfigError(f"watermark config {name} must be a boolean")
    algorithm = raw.get("algorithm")
    if algorithm is not None and algorithm not in WATERMARK_ALGORITHMS:
        raise WatermarkConfigError(
            f"watermark config algorithm must be one of {WATERMARK_ALGORITHMS}"
        )
    synthid_depth = raw.get("synthid_depth")
    if synthid_depth is not None and (
        isinstance(synthid_depth, bool)
        or not isinstance(synthid_depth, int)
        or not 1 <= synthid_depth <= MAX_SYNTHID_DEPTH
    ):
        raise WatermarkConfigError(
            f"watermark config synthid_depth must be an integer from 1 to "
            f"{MAX_SYNTHID_DEPTH}"
        )
    synthid_candidate_cap = raw.get("synthid_candidate_cap")
    if synthid_candidate_cap is not None and (
        isinstance(synthid_candidate_cap, bool)
        or not isinstance(synthid_candidate_cap, int)
        or synthid_candidate_cap < 1
    ):
        raise WatermarkConfigError(
            "watermark config synthid_candidate_cap must be a positive integer"
        )
    if algorithm == "synthid":
        for name in ("key_b", "mixing_probability"):
            if raw.get(name) is not None:
                raise WatermarkConfigError(
                    f"watermark config {name} is not supported with "
                    "algorithm='synthid'"
                )
    return WatermarkServerConfig(**raw)
