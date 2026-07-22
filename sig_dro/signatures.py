from __future__ import annotations

"""Geometric signatures used by the paper experiments."""

from functools import lru_cache
from itertools import product
from typing import Dict, Literal

import numpy as np


Array = np.ndarray
SignatureBackend = Literal["iisignature"]
SignatureType = Literal["geometric"]
SignatureRepresentation = Literal["signature"]


try:  # the environment checker reports a clearer installation error
    import iisignature as _iisignature  # type: ignore
except Exception:  # pragma: no cover
    _iisignature = None


@lru_cache(maxsize=None)
def iisignature_available() -> bool:
    return _iisignature is not None


def resolve_signature_backend(requested_backend: str = "iisignature") -> str:
    if str(requested_backend) != "iisignature":
        raise ValueError("The paper implementation requires signature_backend='iisignature'.")
    if not iisignature_available():
        raise RuntimeError("iisignature is required; install the pinned requirements first.")
    return "iisignature"


def resolve_signature_type(requested_type: str = "geometric") -> str:
    if str(requested_type) != "geometric":
        raise ValueError("The paper implementation uses geometric signatures.")
    return "geometric"


def resolve_signature_representation(requested_representation: str = "signature") -> str:
    if str(requested_representation) != "signature":
        raise ValueError("The paper implementation uses ordinary signatures.")
    return "signature"


def signature_feature_size(path_dim: int, level: int) -> int:
    return int(sum(int(path_dim) ** k for k in range(1, int(level) + 1)))


@lru_cache(maxsize=None)
def _level_offsets(path_dim: int, max_level: int) -> tuple[int, ...]:
    offsets = [0]
    total = 0
    for level in range(1, int(max_level) + 1):
        total += int(path_dim) ** level
        offsets.append(total)
    return tuple(offsets)


@lru_cache(maxsize=None)
def _iisignature_siglength(path_dim: int, max_level: int) -> int:
    if _iisignature is None:
        raise RuntimeError("iisignature is required; install the pinned requirements first.")
    if hasattr(_iisignature, "siglength"):
        return int(_iisignature.siglength(int(path_dim), int(max_level)))
    return signature_feature_size(path_dim, max_level)


def _signature_from_augmented_paths(paths: Array, max_level: int) -> Dict[int, Array]:
    arr = np.asarray(paths, dtype=float)
    max_level = int(max_level)
    if max_level < 1:
        raise ValueError("max_level must be positive.")
    if arr.ndim != 3:
        raise ValueError(f"Expected paths with shape (batch, steps + 1, dim), got {arr.shape}.")
    if _iisignature is None:
        raise RuntimeError("iisignature is required; install the pinned requirements first.")

    batch, _, path_dim = arr.shape
    expected_width = _iisignature_siglength(path_dim, max_level)
    try:
        full = np.asarray(_iisignature.sig(arr, max_level), dtype=float)
        if full.ndim == 1:
            full = full[None, :]
        if full.shape != (batch, expected_width):
            raise ValueError(
                f"Unexpected batched signature shape {full.shape}; expected {(batch, expected_width)}."
            )
    except Exception:
        full = np.empty((batch, expected_width), dtype=float)
        for index in range(batch):
            full[index] = np.asarray(_iisignature.sig(arr[index], max_level), dtype=float)

    offsets = _level_offsets(path_dim, max_level)
    return {level: full[:, : offsets[level]] for level in range(1, max_level + 1)}


def _time_augmented_paths(
    times: Array,
    paths: Array,
    *,
    add_running_max_channel: bool,
    add_running_min_channel: bool,
) -> Array:
    grid = np.asarray(times, dtype=float).reshape(-1)
    state = np.asarray(paths, dtype=float)
    if state.ndim != 3:
        raise ValueError(f"Expected paths with shape (batch, steps + 1, state_dim), got {state.shape}.")
    if grid.shape[0] != state.shape[1]:
        raise ValueError("times and paths are inconsistent.")
    if grid.size == 0 or grid[-1] == 0:
        raise ValueError("times[-1] must be non-zero so that time can be normalized to [0, 1].")

    channels = [np.broadcast_to((grid / grid[-1])[None, :, None], (state.shape[0], state.shape[1], 1)), state]
    if add_running_min_channel:
        channels.append(np.minimum.accumulate(state, axis=1))
    if add_running_max_channel:
        channels.append(np.maximum.accumulate(state, axis=1))
    return np.concatenate(channels, axis=2)


def time_augmented_signatures(
    times: Array,
    paths: Array,
    max_level: int,
    eps: float = 1e-12,
    backend: SignatureBackend = "iisignature",
    return_backend: bool = False,
    add_running_max_channel: bool = False,
    add_running_min_channel: bool = False,
    signature_type: SignatureType = "geometric",
    signature_representation: SignatureRepresentation = "signature",
) -> Dict[int, Array] | tuple[Dict[int, Array], str]:
    """Compute paper signatures for a scalar state path augmented by time."""

    del eps
    resolve_signature_backend(backend)
    resolve_signature_type(signature_type)
    resolve_signature_representation(signature_representation)
    state = np.asarray(paths, dtype=float)
    if state.ndim != 2:
        raise ValueError(f"Expected paths with shape (batch, steps + 1), got {state.shape}.")
    augmented = _time_augmented_paths(
        times,
        state[:, :, None],
        add_running_max_channel=bool(add_running_max_channel),
        add_running_min_channel=bool(add_running_min_channel),
    )
    features = _signature_from_augmented_paths(augmented, max_level=int(max_level))
    return (features, "iisignature") if return_backend else features


def time_augmented_multichannel_signatures(
    times: Array,
    paths: Array,
    max_level: int,
    eps: float = 1e-12,
    backend: SignatureBackend = "iisignature",
    return_backend: bool = False,
    add_running_max_channel: bool = False,
    add_running_min_channel: bool = False,
    signature_type: SignatureType = "geometric",
    signature_representation: SignatureRepresentation = "signature",
) -> Dict[int, Array] | tuple[Dict[int, Array], str]:
    """Compute paper signatures for a multichannel state path augmented by time."""

    del eps
    resolve_signature_backend(backend)
    resolve_signature_type(signature_type)
    resolve_signature_representation(signature_representation)
    augmented = _time_augmented_paths(
        times,
        np.asarray(paths, dtype=float),
        add_running_max_channel=bool(add_running_max_channel),
        add_running_min_channel=bool(add_running_min_channel),
    )
    features = _signature_from_augmented_paths(augmented, max_level=int(max_level))
    return (features, "iisignature") if return_backend else features


def signature_coordinate_names(
    path_dim: int,
    max_level: int,
    channel_names: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Return signature words in the level-by-level order used by iisignature."""

    path_dim = int(path_dim)
    max_level = int(max_level)
    if path_dim <= 0 or max_level <= 0:
        raise ValueError("path_dim and max_level must be positive.")
    if channel_names is None:
        names = tuple(f"c{index}" for index in range(path_dim))
    else:
        names = tuple(str(name) for name in channel_names)
        if len(names) != path_dim:
            raise ValueError("channel_names length must equal path_dim.")
    return [
        "".join(word)
        for level in range(1, max_level + 1)
        for word in product(names, repeat=level)
    ]
