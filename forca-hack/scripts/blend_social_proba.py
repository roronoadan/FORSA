from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_meta(meta_path: Path) -> list[int]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return [int(x) for x in meta["classes"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", type=str, required=True, help="Test CSV (to get ids).")
    ap.add_argument("--proba", type=str, nargs="+", required=True, help="One or more .npy probability files.")
    ap.add_argument("--meta", type=str, nargs="+", required=True, help="Matching meta .json files with class order.")
    ap.add_argument("--weights", type=float, nargs="*", default=None, help="Optional weights for each proba file.")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    proba_paths = [Path(p) for p in args.proba]
    meta_paths = [Path(p) for p in args.meta]
    if len(proba_paths) != len(meta_paths):
        raise ValueError("Must provide the same number of --proba and --meta files.")

    weights = args.weights
    if weights is None or len(weights) == 0:
        weights = [1.0] * len(proba_paths)
    if len(weights) != len(proba_paths):
        raise ValueError("If provided, --weights must match number of --proba files.")

    # Load first classes as canonical
    classes = load_meta(meta_paths[0])
    num_classes = len(classes)

    proba_sum = None
    wsum = 0.0
    for p_path, m_path, w in zip(proba_paths, meta_paths, weights, strict=True):
        cls = load_meta(m_path)
        if cls != classes:
            raise ValueError(f"Class order mismatch between meta files. {m_path} != {meta_paths[0]}")
        p = np.load(p_path)
        if p.shape[1] != num_classes:
            raise ValueError(f"Bad shape {p.shape} for {p_path}; expected (*,{num_classes})")
        if proba_sum is None:
            proba_sum = np.zeros_like(p, dtype=np.float64)
        proba_sum += float(w) * p.astype(np.float64)
        wsum += float(w)

    assert proba_sum is not None
    proba = proba_sum / max(wsum, 1e-12)
    pred_idx = np.argmax(proba, axis=1)
    pred_label = np.array([classes[int(i)] for i in pred_idx], dtype=np.int64)

    test_df = pd.read_csv(args.test_csv)
    if "id" not in test_df.columns:
        raise KeyError("Expected 'id' column in test_csv")

    out = pd.DataFrame({"id": test_df["id"].astype(int), "Class": pred_label.astype(int)})
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved blended submission: {out_path}")


if __name__ == "__main__":
    main()


