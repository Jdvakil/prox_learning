#!/usr/bin/env python3
"""Deterministic V10 survivor catalogs without expanded assembly_id strings.

The v1 catalog stores an object `assembly_ids` array that expands to ~5.37 GB.
This module never materializes that member. Topology and assembly IDs are
derived lazily from numeric lobe keys. Streaming loaders memmap key arrays.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from pact_place_corridor_contract import sha256_file
from pact_place_v10_compound_pendant_contract import (
    CATALOG_SCHEMA_V2,
    STEM_HALF_M,
    STEM_TOP_Z_M,
    CEILING_TOP_Z_M,
    CROSSBAR_HEIGHT_M,
    round_m,
)
from pact_place_v10_geometry import assembly_id_for, union_aabb_key

ROOT = Path(__file__).resolve().parents[1]


def assembly_id_from_lobe_keys(
    lobe_keys: Sequence[Sequence[float]],
    *,
    topology: str | None = None,
) -> str:
    keys = [tuple(round_m(float(value)) for value in row) for row in lobe_keys]
    keys = [key for key in keys if not any(np.isnan(value) for value in key)]
    n_lobes = len(keys)
    if topology is None:
        topology = "two_lobe" if n_lobes == 2 else "three_lobe"
    lobes = [{"key": list(key)} for key in keys]
    return assembly_id_for(topology, lobes)


def _read_npy_from_zip(path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def load_prefilter_lobe_keys(path: Path) -> np.ndarray:
    """Load only numeric lobe keys from the superseded v1 catalog. No assembly_ids."""
    names = zipfile.ZipFile(path).namelist()
    if "assembly_ids.npy" not in names:
        raise ValueError(f"{path} is missing assembly_ids.npy; not a v1 catalog")
    keys = _read_npy_from_zip(path, "lobe_keys.npy")
    if keys.ndim != 3 or keys.shape[1] < 2 or keys.shape[2] != 6:
        raise ValueError(f"unexpected lobe_keys shape {keys.shape}")
    return np.asarray(keys[:, :2, :], dtype=np.float64)


def load_prefilter_volumes(path: Path) -> np.ndarray:
    return np.asarray(_read_npy_from_zip(path, "volume_m3.npy"), dtype=np.float64)


def load_prefilter_bits(path: Path) -> np.ndarray:
    return np.asarray(
        _read_npy_from_zip(path, "lobe_necessity_bits.npy"), dtype=np.int32
    )


def load_prefilter_margins(path: Path) -> np.ndarray:
    return np.asarray(
        _read_npy_from_zip(path, "min_grasp_clearance_margin_m.npy"),
        dtype=np.float64,
    )


def prefilter_row_count(path: Path) -> int:
    bits = load_prefilter_bits(path)
    return int(bits.shape[0])


def lexsort_two_lobe_keys(keys: np.ndarray) -> np.ndarray:
    packed = np.asarray(keys, dtype=np.float64).reshape(len(keys), -1)
    return np.lexsort(packed.T[::-1])


def write_survivor_catalog_v2(
    path: Path,
    *,
    lobe_keys: np.ndarray,
    volume_m3: np.ndarray,
    lobe_necessity_bits: np.ndarray,
    min_grasp_clearance_margin_m: np.ndarray,
    topology: str = "two_lobe",
) -> str:
    keys = np.asarray(lobe_keys, dtype=np.float64)
    if keys.ndim != 3 or keys.shape[2] != 6:
        raise ValueError(f"lobe_keys must be (N, n_lobes, 6), got {keys.shape}")
    n_rows = int(keys.shape[0])
    n_lobes = int(keys.shape[1])
    order = lexsort_two_lobe_keys(keys) if n_rows else np.arange(0, dtype=np.int64)
    keys = keys[order]
    volume = np.asarray(volume_m3, dtype=np.float64)[order]
    bits = np.asarray(lobe_necessity_bits, dtype=np.int32)[order]
    margin = np.asarray(min_grasp_clearance_margin_m, dtype=np.float64)[order]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=np.asarray(CATALOG_SCHEMA_V2),
        topology=np.asarray(topology),
        n_rows=np.asarray(n_rows, dtype=np.int64),
        n_lobes=np.asarray(n_lobes, dtype=np.int32),
        lobe_keys=keys,
        volume_m3=volume,
        lobe_necessity_bits=bits,
        min_grasp_clearance_margin_m=margin,
    )
    return sha256_file(path)


class SurvivorCatalogV2:
    """Streaming catalog. Does not keep assembly_id strings in memory."""

    _ARRAY_MEMBERS = (
        "lobe_keys.npy",
        "volume_m3.npy",
        "lobe_necessity_bits.npy",
        "min_grasp_clearance_margin_m.npy",
    )

    def __init__(self, path: Path, *, mmap: bool = True) -> None:
        self.path = Path(path)
        names = zipfile.ZipFile(self.path).namelist()
        if "assembly_ids.npy" in names:
            raise ValueError(
                f"{self.path} contains assembly_ids.npy; refuse to materialize it"
            )
        if mmap:
            arrays = {member: self._mmap_member(member) for member in self._ARRAY_MEMBERS}
            self.lobe_keys = arrays["lobe_keys.npy"]
            self.volume_m3 = arrays["volume_m3.npy"]
            self.lobe_necessity_bits = arrays["lobe_necessity_bits.npy"]
            self.min_grasp_clearance_margin_m = arrays["min_grasp_clearance_margin_m.npy"]
        else:
            self.lobe_keys = _read_npy_from_zip(self.path, "lobe_keys.npy")
            self.volume_m3 = _read_npy_from_zip(self.path, "volume_m3.npy")
            self.lobe_necessity_bits = _read_npy_from_zip(
                self.path, "lobe_necessity_bits.npy"
            )
            self.min_grasp_clearance_margin_m = _read_npy_from_zip(
                self.path, "min_grasp_clearance_margin_m.npy"
            )
        self.topology = str(_read_npy_from_zip(self.path, "topology.npy"))
        self.n_lobes = int(_read_npy_from_zip(self.path, "n_lobes.npy"))
        self.n_rows = int(_read_npy_from_zip(self.path, "n_rows.npy"))

    def _mmap_dir(self) -> Path:
        return self.path.parent / (self.path.stem + "_mmap")

    def _mmap_member(self, member: str) -> np.ndarray:
        dest = self._mmap_dir() / member
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(member) as handle:
                    dest.write_bytes(handle.read())
        return np.load(dest, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.n_rows)

    def row(self, index: int) -> dict[str, Any]:
        keys = np.asarray(self.lobe_keys[int(index)], dtype=np.float64)
        return {
            "index": int(index),
            "topology": self.topology,
            "n_lobes": int(self.n_lobes),
            "lobe_keys": keys,
            "volume_m3": float(self.volume_m3[int(index)]),
            "lobe_necessity_bits": int(self.lobe_necessity_bits[int(index)]),
            "min_grasp_clearance_margin_m": float(
                self.min_grasp_clearance_margin_m[int(index)]
            ),
            "assembly_id": assembly_id_from_lobe_keys(keys, topology=self.topology),
        }

    def iter_rows(self, *, batch: int = 4096) -> Iterator[dict[str, Any]]:
        for start in range(0, self.n_rows, int(batch)):
            stop = min(self.n_rows, start + int(batch))
            for index in range(start, stop):
                yield self.row(index)

    def memory_bound_fields(self) -> tuple[str, ...]:
        return ("lobe_keys", "volume_m3", "lobe_necessity_bits", "min_grasp_clearance_margin_m")


def stem_center_half_from_lobe_key(
    key: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz, hx, hy, hz = (float(value) for value in key)
    outward = -1.0 if cy < 0.0 else 1.0
    stem_y = cy + outward * hy
    lobe_top = cz + hz
    half_z = (STEM_TOP_Z_M - lobe_top) / 2.0
    center = np.asarray([cx, stem_y, lobe_top + half_z], dtype=np.float64)
    half = np.asarray([STEM_HALF_M, STEM_HALF_M, half_z], dtype=np.float64)
    return center, half


def crossbar_center_half_from_stem_ys(
    center_x: float, stem_y_a: float, stem_y_b: float
) -> tuple[np.ndarray, np.ndarray]:
    y_lo = min(float(stem_y_a), float(stem_y_b)) - STEM_HALF_M
    y_hi = max(float(stem_y_a), float(stem_y_b)) + STEM_HALF_M
    half_z = CROSSBAR_HEIGHT_M / 2.0
    center = np.asarray(
        [float(center_x), 0.5 * (y_lo + y_hi), CEILING_TOP_Z_M - half_z],
        dtype=np.float64,
    )
    half = np.asarray([STEM_HALF_M, 0.5 * (y_hi - y_lo), half_z], dtype=np.float64)
    return center, half


def union_aabb_from_two_lobe_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized union AABB for two-lobe rows. keys shape (N, 2, 6)."""
    packed = np.asarray(keys, dtype=np.float64)
    k0 = packed[:, 0, :]
    k1 = packed[:, 1, :]
    lo0 = k0[:, :3] - k0[:, 3:]
    hi0 = k0[:, :3] + k0[:, 3:]
    lo1 = k1[:, :3] - k1[:, 3:]
    hi1 = k1[:, :3] + k1[:, 3:]
    n = packed.shape[0]
    stem_y0 = k0[:, 1] + np.where(k0[:, 1] < 0.0, -1.0, 1.0) * k0[:, 4]
    stem_y1 = k1[:, 1] + np.where(k1[:, 1] < 0.0, -1.0, 1.0) * k1[:, 4]
    lobe_top0 = k0[:, 2] + k0[:, 5]
    lobe_top1 = k1[:, 2] + k1[:, 5]
    stem_hz0 = (STEM_TOP_Z_M - lobe_top0) / 2.0
    stem_hz1 = (STEM_TOP_Z_M - lobe_top1) / 2.0
    stem_cz0 = lobe_top0 + stem_hz0
    stem_cz1 = lobe_top1 + stem_hz1
    stem_lo0 = np.stack(
        [k0[:, 0] - STEM_HALF_M, stem_y0 - STEM_HALF_M, stem_cz0 - stem_hz0], axis=1
    )
    stem_hi0 = np.stack(
        [k0[:, 0] + STEM_HALF_M, stem_y0 + STEM_HALF_M, stem_cz0 + stem_hz0], axis=1
    )
    stem_lo1 = np.stack(
        [k1[:, 0] - STEM_HALF_M, stem_y1 - STEM_HALF_M, stem_cz1 - stem_hz1], axis=1
    )
    stem_hi1 = np.stack(
        [k1[:, 0] + STEM_HALF_M, stem_y1 + STEM_HALF_M, stem_cz1 + stem_hz1], axis=1
    )
    y_lo = np.minimum(stem_y0, stem_y1) - STEM_HALF_M
    y_hi = np.maximum(stem_y0, stem_y1) + STEM_HALF_M
    half_z = CROSSBAR_HEIGHT_M / 2.0
    bar_lo = np.stack(
        [k0[:, 0] - STEM_HALF_M, y_lo, np.full(n, CEILING_TOP_Z_M - CROSSBAR_HEIGHT_M)],
        axis=1,
    )
    bar_hi = np.stack(
        [k0[:, 0] + STEM_HALF_M, y_hi, np.full(n, CEILING_TOP_Z_M)],
        axis=1,
    )
    union_lo = np.minimum.reduce([lo0, lo1, stem_lo0, stem_lo1, bar_lo])
    union_hi = np.maximum.reduce([hi0, hi1, stem_hi0, stem_hi1, bar_hi])
    return union_lo, union_hi


def unique_union_count(keys: np.ndarray, *, ndigits: int = 9) -> int:
    lo, hi = union_aabb_from_two_lobe_keys(keys)
    packed = np.round(np.concatenate([lo, hi], axis=1), ndigits)
    return int(np.unique(packed, axis=0).shape[0])


def union_key_from_two_lobe_key(key: Sequence[Sequence[float]]) -> tuple[float, ...]:
    lo, hi = union_aabb_from_two_lobe_keys(np.asarray(key, dtype=np.float64).reshape(1, 2, 6))
    return union_aabb_key(lo[0], hi[0])


def stem_keys_from_lobe_keys(keys: np.ndarray) -> np.ndarray:
    """Vectorized stem center+half keys. keys shape (N, 2, 6) -> (N, 2, 6)."""
    packed = np.asarray(keys, dtype=np.float64)
    stem_y = packed[:, :, 1] + np.where(packed[:, :, 1] < 0.0, -1.0, 1.0) * packed[:, :, 4]
    lobe_top = packed[:, :, 2] + packed[:, :, 5]
    half_z = (STEM_TOP_Z_M - lobe_top) / 2.0
    center_z = lobe_top + half_z
    hx = np.full(packed.shape[:2], STEM_HALF_M, dtype=np.float64)
    return np.stack(
        [packed[:, :, 0], stem_y, center_z, hx, hx, half_z],
        axis=2,
    )


def crossbar_keys_from_lobe_keys(keys: np.ndarray) -> np.ndarray:
    """Vectorized crossbar center+half keys. keys shape (N, 2, 6) -> (N, 6)."""
    packed = np.asarray(keys, dtype=np.float64)
    stem_y0 = packed[:, 0, 1] + np.where(packed[:, 0, 1] < 0.0, -1.0, 1.0) * packed[:, 0, 4]
    stem_y1 = packed[:, 1, 1] + np.where(packed[:, 1, 1] < 0.0, -1.0, 1.0) * packed[:, 1, 4]
    y_lo = np.minimum(stem_y0, stem_y1) - STEM_HALF_M
    y_hi = np.maximum(stem_y0, stem_y1) + STEM_HALF_M
    half_z = CROSSBAR_HEIGHT_M / 2.0
    return np.stack(
        [
            packed[:, 0, 0],
            0.5 * (y_lo + y_hi),
            np.full(packed.shape[0], CEILING_TOP_Z_M - half_z),
            np.full(packed.shape[0], STEM_HALF_M),
            0.5 * (y_hi - y_lo),
            np.full(packed.shape[0], half_z),
        ],
        axis=1,
    )


def unique_rounded_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed = np.round(np.asarray(keys, dtype=np.float64).reshape(-1, 6), 9)
    unique, inverse = np.unique(packed, axis=0, return_inverse=True)
    return unique, inverse.astype(np.int32)


def assembly_from_two_lobe_keys(
    keys: Sequence[Sequence[float]],
    *,
    aperture_width_m: float | None = None,
) -> dict[str, Any]:
    from pact_place_v10_compound_pendant_contract import DEFAULT_APERTURE_WIDTH_M
    from pact_place_v10_geometry import build_assembly, lobe_from_key

    width = DEFAULT_APERTURE_WIDTH_M if aperture_width_m is None else float(aperture_width_m)
    lobes = [lobe_from_key(row, aperture_width_m=width) for row in keys]
    return build_assembly(lobes, aperture_width_m=width)


def verify_prefilter_catalog(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(
            f"prefilter catalog SHA mismatch: got {digest}, expected {expected_sha256}"
        )
    names = zipfile.ZipFile(path).namelist()
    if "lobe_keys.npy" not in names:
        raise RuntimeError(f"{path} is missing lobe_keys.npy")
    return {"path": str(path), "sha256": digest, "zip_members": sorted(names)}
