"""Utilities to inspect callables and adapt keyword arguments."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

__all__ = ["filter_kwargs"]


def filter_kwargs(function: Callable[..., Any], options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``options`` containing only parameters accepted by ``function``.

    The helper inspects the signature of ``function`` to determine which keyword arguments
    are supported. When the callable accepts arbitrary keyword arguments (``**kwargs``)
    the original mapping is returned unchanged. If the signature cannot be inspected the
    options are forwarded as-is.
    """

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return dict(options)

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return dict(options)

    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    return {name: value for name, value in options.items() if name in allowed}
