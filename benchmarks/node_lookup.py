# Copyright (C) 2026 Mads Kruse
#
# This file is part of shapemesh.
#
# shapemesh is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# shapemesh is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with shapemesh. If not, see <https://www.gnu.org/licenses/>.

# ruff: noqa: T201
"""Benchmark node-tag lookup strategies without imposing CI timing gates.

The default matrix uses 1e3, 1e5, 1e6, and 1e7 nodes and queries of 1, 1e3,
and 1e6 tags. Use ``--quick`` for a small local smoke run. Cached auxiliary
array bytes are reported directly because tracemalloc does not reliably track
NumPy allocations on every platform.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from time import perf_counter

import numpy as np

from shapemesh import NodeTable

LAYOUTS = (
    "ascending-contiguous",
    "unsorted-contiguous",
    "mildly-sparse",
    "highly-sparse",
    "random-order",
)
FULL_SIZES = (1_000, 100_000, 1_000_000, 10_000_000)
FULL_QUERY_SIZES = (1, 1_000, 1_000_000)


def _make_tags(layout: str, size: int) -> np.ndarray:
    rng = np.random.default_rng(20261031)
    if layout == "ascending-contiguous":
        return np.arange(1, size + 1, dtype=np.int64)
    if layout == "unsorted-contiguous":
        tags = np.arange(1, size + 1, dtype=np.int64)
        rng.shuffle(tags)
        return tags
    if layout == "mildly-sparse":
        return np.arange(1, 2 * size + 1, 2, dtype=np.int64)
    if layout == "highly-sparse":
        return np.arange(1, 1_000_000 * size + 1, 1_000_000, dtype=np.int64)
    if layout == "random-order":
        tags = np.arange(1, 1_000 * size + 1, 1_000, dtype=np.int64)
        rng.shuffle(tags)
        return tags
    raise ValueError(f"unknown layout: {layout}")


def _cached_nbytes(nodes: NodeTable) -> int:
    lookup = nodes._lookup
    if lookup is None:
        return 0
    arrays = (lookup.direct, lookup.sorted_tags, lookup.order)
    return sum(
        int(array.nbytes) for array in arrays if array is not None and array is not nodes.tags
    )


def _best_time(call: Callable[[], object], repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = perf_counter()
        call()
        best = min(best, perf_counter() - start)
    return best


def _benchmark_case(layout: str, size: int, query_sizes: tuple[int, ...]) -> None:
    tags = _make_tags(layout, size)
    nodes = NodeTable(tags, np.zeros((size, 3), dtype=np.float64))
    scalar_tag = int(tags[size // 2])

    start = perf_counter()
    nodes.indices_for(scalar_tag)
    construction_ms = (perf_counter() - start) * 1e3
    lookup = nodes._lookup
    assert lookup is not None
    strategy = lookup.strategy
    cache_mib = _cached_nbytes(nodes) / 1024**2

    scalar_repeats = 100
    scalar_indices_us = _best_time(lambda: nodes.indices_for(scalar_tag), scalar_repeats) * 1e6
    scalar_coords_us = _best_time(lambda: nodes.coords_for(scalar_tag), scalar_repeats) * 1e6
    print(
        f"{layout:22} {size:9d} {strategy:10} {construction_ms:11.3f} "
        f"{cache_mib:10.3f} {'scalar':>6} {1:9d} "
        f"{scalar_indices_us:12.3f} {scalar_coords_us:11.3f}"
    )

    rng = np.random.default_rng(20261101)
    for query_size in query_sizes:
        query = tags[rng.integers(0, size, size=query_size)]
        repeats = 5 if query_size <= 1_000 else 2
        indices_us = _best_time(partial(nodes.indices_for, query), repeats) * 1e6
        coords_us = _best_time(partial(nodes.coords_for, query), repeats) * 1e6
        print(
            f"{layout:22} {size:9d} {strategy:10} {construction_ms:11.3f} "
            f"{cache_mib:10.3f} {'array':>6} {query_size:9d} "
            f"{indices_us:12.3f} {coords_us:11.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only 1e3 and 1e5 nodes with at most 1e3 query tags",
    )
    args = parser.parse_args()
    sizes = (1_000, 100_000) if args.quick else FULL_SIZES
    query_sizes = (1, 1_000) if args.quick else FULL_QUERY_SIZES

    print(
        f"{'layout':22} {'nodes':>9} {'strategy':10} {'build ms':>11} "
        f"{'cache MiB':>10} {'query':>6} {'queries':>9} "
        f"{'indices us':>12} {'coords us':>11}"
    )
    for layout in LAYOUTS:
        for size in sizes:
            _benchmark_case(layout, size, query_sizes)


if __name__ == "__main__":
    main()
