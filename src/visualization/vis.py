#!/usr/bin/env python3
"""
Script version of `vis.ipynb` — plot an MRC image with particle annotations.

Usage example:
    python src/visualization/vis.py --mrc /path/to/stack_0002_2x_SumCorr.mrc \\
        --csv /path/to/stack_0002_2x_SumCorr.csv
"""

import argparse
import warnings
import os
import time
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mrcfile


def plot_annot_mrc(mrc_file_path, particle_annotation):
    if not os.path.exists(mrc_file_path):
        raise FileNotFoundError(f"MRC file not found: {mrc_file_path}")
    if not os.path.exists(particle_annotation):
        raise FileNotFoundError(
            f"Annotation file not found: {particle_annotation}")

    mrc_data = mrcfile.read(mrc_file_path)
    annot = pd.read_csv(particle_annotation)

    plt.imshow(mrc_data, cmap="gray")
    plt.grid(False)

    if "Y-Coordinate" not in annot.columns or "X-Coordinate" not in annot.columns:
        raise ValueError(
            "Annotation CSV must contain 'X-Coordinate' and 'Y-Coordinate' columns"
        )

    # Adjust Y for image coordinate system used by imshow
    annot["Y_fixed"] = mrc_data.shape[0] - annot["Y-Coordinate"] - 1
    plt.scatter(annot["X-Coordinate"], annot["Y_fixed"], c="green", s=10)
    plt.title(os.path.basename(mrc_file_path))
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot MRC with particle annotations")
    parser.add_argument(
        "--mrc",
        help="Path to the MRC file",
        default=(
            "/Users/lucindashaffer/Documents/10005/micrographs/stack_0002_2x_SumCorr.mrc"
        ),
    )
    parser.add_argument(
        "--csv",
        help="Path to the annotation CSV",
        default=(
            "/Users/lucindashaffer/Documents/10005/ground_truth/particle_coordinates/stack_0002_2x_SumCorr.csv"
        ),
    )
    parser.add_argument(
        "--bench-size",
        type=int,
        default=0,
        help="If >0 run hash-table benchmark with this many items",
    )
    parser.add_argument(
        "--bench-only",
        action="store_true",
        help="Run benchmark only and exit (do not plot)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (used by benchmark)",
    )

    args = parser.parse_args()

    # Notebook-like visual defaults
    sns.set(style="whitegrid", context="notebook", palette="deep")
    warnings.filterwarnings("ignore")
    plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 100})
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_rows", 100)

    # If requested, run benchmark and optionally exit early
    if getattr(args, "bench_size", 0) and args.bench_size > 0:
        benchmark_hash_structures(n=args.bench_size, seed=args.seed)
        if args.bench_only:
            return

    plot_annot_mrc(args.mrc, args.csv)


class HashTable:
    """
    A minimal separate-chaining hash table for benchmarking/demonstration.
    Not intended to be feature-complete — simple insert/get semantics only.
    """

    def __init__(self, capacity=20011):
        # use a prime-like default capacity for nicer distribution
        self._buckets = [[] for _ in range(capacity)]
        self._capacity = capacity

    def _bucket_index(self, key):
        return hash(key) % self._capacity

    def insert(self, key, value):
        idx = self._bucket_index(key)
        bucket = self._buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))

    def get(self, key, default=None):
        idx = self._bucket_index(key)
        bucket = self._buckets[idx]
        for k, v in bucket:
            if k == key:
                return v
        return default


def benchmark_hash_structures(n=10000, seed=42):
    """Compare insert + lookup times for custom HashTable vs Python dict.
    Prints durations and a simple comparison summary (custom vs dict).
    """
    
    random.seed(seed)
    keys = [random.randint(0, 10**9) for _ in range(n)]
    values = [random.random() for _ in range(n)]

    # Benchmark custom HashTable
    ht = HashTable(capacity=max(3, n * 2 + 1))
    t0 = time.perf_counter()
    for k, v in zip(keys, values):
        ht.insert(k, v)
    t1 = time.perf_counter()
    for k in keys:
        _ = ht.get(k)
    t2 = time.perf_counter()
    ht_insert_time = t1 - t0
    ht_lookup_time = t2 - t1

    # Benchmark Python dict
    d = {}
    t0 = time.perf_counter()
    for k, v in zip(keys, values):
        d[k] = v
    t1 = time.perf_counter()
    for k in keys:
        _ = d.get(k)
    t2 = time.perf_counter()
    dict_insert_time = t1 - t0
    dict_lookup_time = t2 - t1

    print("\nHash table benchmark (n={})".format(n))
    print("Custom HashTable: insert = {:.6f}s, lookup = {:.6f}s".format(
        ht_insert_time, ht_lookup_time))
    print("Python dict     : insert = {:.6f}s, lookup = {:.6f}s".format(
        dict_insert_time, dict_lookup_time))

    # Comparison (before vs after style): custom vs dict
    def ratio(a, b):
        return a / b if b else float('inf')

    print("\nComparison (custom / dict):")
    print(" insert ratio = {:.3f}x".format(
        ratio(ht_insert_time, dict_insert_time)))
    print(" lookup ratio = {:.3f}x".format(
        ratio(ht_lookup_time, dict_lookup_time)))


if __name__ == "__main__":
    main()
