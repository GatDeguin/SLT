"""Utilities for working with metadata fields."""
from __future__ import annotations

import math
from typing import Optional


def sanitize_time_value(value: str) -> Optional[float]:
    """Normalise a numeric time value coming from metadata sources."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result

    text = str(value).strip()
    if not text:
        return None

    normalised = text.replace(",", ".").replace(" ", "")

    if normalised.count(".") > 1:
        parts = normalised.split(".")
        integer = parts[0]
        decimals = "".join(parts[1:])
        normalised = f"{integer}.{decimals}" if decimals else integer

    try:
        result = float(normalised)
    except ValueError:
        return None

    if math.isnan(result) or math.isinf(result):
        return None
    return result
