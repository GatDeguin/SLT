"""Parsers utilitarios para argumentos de línea de comandos."""

from __future__ import annotations

import re
from typing import Tuple


_DEFAULT_SPLIT_PATTERN = re.compile(r"[\s,]+")


def parse_float_sequence(text: str) -> Tuple[float, ...]:
    """Convierte ``text`` en una tupla de ``float`` separando por coma o espacio."""

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("el valor no puede estar vacío")
    if cleaned.lower() in {"none", "null", "false"}:
        raise ValueError("no se puede convertir 'none' en una secuencia numérica")
    cleaned = cleaned.replace(";", ",")
    parts = [part for part in _DEFAULT_SPLIT_PATTERN.split(cleaned) if part]
    if not parts:
        raise ValueError("no se encontraron números en el valor proporcionado")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:  # pragma: no cover - mensaje detallado en error.
        raise ValueError(f"no se pudo convertir '{text}' en números flotantes") from exc


def parse_range_pair(
    text: str,
    *,
    positive: bool = False,
    symmetric_single: bool = False,
) -> Tuple[float, float]:
    """Parsea un rango ``min,max`` devolviendo un par ordenado."""

    values = parse_float_sequence(text)
    if len(values) == 1:
        value = values[0]
        low = -value if symmetric_single else value
        high = value
    elif len(values) == 2:
        low, high = values
    else:
        raise ValueError("el rango debe contener exactamente 1 o 2 valores")
    if low > high:
        raise ValueError("el mínimo del rango no puede ser mayor al máximo")
    if positive and (low <= 0 or high <= 0):
        raise ValueError("el rango debe contener valores mayores a cero")
    return float(low), float(high)


def parse_translation_range(text: str) -> Tuple[float, float, float, float]:
    """Interpreta traslaciones ``(x, y)`` admitiendo 1, 2 o 4 valores."""

    values = parse_float_sequence(text)
    if len(values) == 1:
        delta = values[0]
        return (-delta, delta, -delta, delta)
    if len(values) == 2:
        low, high = values
        if low > high:
            raise ValueError("el mínimo del rango no puede ser mayor al máximo")
        return float(low), float(high), float(low), float(high)
    if len(values) == 4:
        min_x, max_x, min_y, max_y = values
        if min_x > max_x or min_y > max_y:
            raise ValueError("cada eje debe cumplir min <= max")
        return float(min_x), float(max_x), float(min_y), float(max_y)
    raise ValueError("se requieren 1, 2 o 4 valores para la traslación")


__all__ = [
    "parse_float_sequence",
    "parse_range_pair",
    "parse_translation_range",
]
