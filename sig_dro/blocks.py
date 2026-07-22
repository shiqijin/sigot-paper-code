from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Array = np.ndarray
BlockNorm = Literal["l2"]
BlockGrouping = Literal["coordinate"]


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    start: int
    stop: int
    norm: BlockNorm = "l2"

    @property
    def slc(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def width(self) -> int:
        return int(self.stop - self.start)


@dataclass(frozen=True)
class BlockSpec:
    blocks: tuple[FeatureBlock, ...]

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def feature_dim(self) -> int:
        return sum(block.width for block in self.blocks)


def coordinate_block_spec(feature_dim: int, norm: BlockNorm = "l2") -> BlockSpec:
    feature_dim = int(feature_dim)
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    if norm != "l2":
        raise ValueError("The paper implementation uses the l2 block norm.")
    return BlockSpec(tuple(FeatureBlock(name=f"component_{j}", start=j, stop=j + 1, norm=norm) for j in range(feature_dim)))


def build_block_spec(
    feature_dim: int,
    grouping: BlockGrouping = "coordinate",
    norm: BlockNorm = "l2",
    cumulative_dims: Sequence[int] | None = None,
) -> BlockSpec:
    if grouping != "coordinate":
        raise ValueError("The paper implementation uses coordinate blocks.")
    return coordinate_block_spec(feature_dim=feature_dim, norm=norm)


def expand_block_values(values: Sequence[float], block_spec: BlockSpec, feature_dim: int | None = None) -> Array:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < block_spec.n_blocks:
        raise ValueError("Budget vector shorter than number of blocks.")
    arr = arr[: block_spec.n_blocks]
    out = np.empty(block_spec.feature_dim, dtype=float)
    for value, block in zip(arr, block_spec.blocks):
        out[block.slc] = float(value)
    if feature_dim is not None and int(feature_dim) != out.shape[0]:
        raise ValueError("feature_dim inconsistent with block_spec.")
    return out


def block_dual_norm(vec: Sequence[float], norm: BlockNorm) -> float:
    x = np.asarray(vec, dtype=float).reshape(-1)
    if norm == "l2":
        return float(np.linalg.norm(x, ord=2))
    raise ValueError("The paper implementation uses the l2 block norm.")
