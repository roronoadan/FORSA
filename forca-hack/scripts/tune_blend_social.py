from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_meta(meta_path: Path) -> list[int]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return [int(x) for x in meta["classes"]]


def align_proba(proba: np.ndarray, src_classes: list[int], tgt_classes: list[int]) -> np.ndarray:
    if src_classes == tgt_classes:
        return proba
    src_idx = {c: i for i, c in enumerate(src_classes)}
    out = np.zeros((proba.shape[0], len(tgt_classes)), dtype=proba.dtype)
    for j, c in enumerate(tgt_classes):
        out[:, j] = proba[:, src_idx[c]]
    return out


def iter_weights(n_models: int, step: float) -> list[list[float]]:
    if n_models == 1:
        return [[1.0]]
    if n_models == 2:
        ws = []
        for w0 in np.arange(0.0, 1.0 + 1e-9, step):
            ws.append([float(w0), float(1.0 - w0)])
        return ws
    if n_models == 3:
        ws = []
        grid = np.arange(0.0, 1.0 + 1e-9, step)
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9:
                    continue
                if w2 < 0:
                    w2 = 0.0
                ws.append([float(w0), float(w1), float(w2)])
        return ws
    raise ValueError("Weight search only supports up to 3 models (use manual weights for more).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--oof_proba", type=str, nargs="+", required=True)
    ap.add_argument("--oof_meta", type=str, nargs="+", required=True)
    ap.add_argument("--test_proba", type=str, nargs="+", required=True)
    ap.add_argument("--test_meta", type=str, nargs="+", required=True)
    ap.add_argument("--step", type=float, default=0.05, help="Grid step for weight search (2 models: 0.02-0.05 is typical).")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_weights_json", type=str, default=None)
    args = ap.parse_args()

    oof_paths = [Path(p) for p in args.oof_proba]
    oof_meta_paths = [Path(p) for p in args.oof_meta]
    test_paths = [Path(p) for p in args.test_proba]
    test_meta_paths = [Path(p) for p in args.test_meta]

    if not (len(oof_paths) == len(oof_meta_paths) == len(test_paths) == len(test_meta_paths)):
        raise ValueError("Counts of oof_proba/oof_meta/test_proba/test_meta must match.")

    n_models = len(oof_paths)
    if n_models not in (1, 2, 3):
        raise ValueError("This script supports 1-3 models.")

    # Check all files exist before proceeding
    all_paths = list(oof_paths) + list(oof_meta_paths) + list(test_paths) + list(test_meta_paths)
    missing = [p for p in all_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files:\n" + "\n".join(f"  - {p}" for p in missing) +
            f"\n\n💡 TIP: Run training scripts with --save_oof_proba and --save_test_proba flags first.\n"
            f"   Example:\n"
            f"   python forca-hack/scripts/train_social_tfidf.py ... --save_oof_proba --save_test_proba\n"
            f"   python forca-hack/scripts/train_social_transformer.py ... --save_oof_proba --save_test_proba"
        )

    # Canonical class order from first meta
    classes = load_meta(oof_meta_paths[0])
    num_classes = len(classes)

    train_df = pd.read_csv(args.train_csv)
    y_true = train_df["Class"].astype(int).to_numpy()
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([label_to_idx[int(v)] for v in y_true], dtype=np.int64)

    oof_list = []
    test_list = []
    for op, om, tp, tm in zip(oof_paths, oof_meta_paths, test_paths, test_meta_paths, strict=True):
        oof = np.load(op)
        test = np.load(tp)
        if oof.shape[1] != num_classes or test.shape[1] != num_classes:
            raise ValueError(f"Shape mismatch: {op} {oof.shape} / {tp} {test.shape} expected (*,{num_classes})")
        oof_cls = load_meta(om)
        test_cls = load_meta(tm)
        oof_list.append(align_proba(oof, oof_cls, classes).astype(np.float64))
        test_list.append(align_proba(test, test_cls, classes).astype(np.float64))

    best_f1 = -1.0
    best_w: list[float] | None = None

    for w in iter_weights(n_models, float(args.step)):
        blend = np.zeros_like(oof_list[0])
        for wi, pi in zip(w, oof_list, strict=True):
            blend += float(wi) * pi
        pred = np.argmax(blend, axis=1)
        f1 = float(f1_score(y_idx, pred, average="macro"))
        if f1 > best_f1:
            best_f1 = f1
            best_w = w

    assert best_w is not None
    print(f"Best OOF macro_f1={best_f1:.5f} weights={best_w} classes={classes}")

    # Blend test with best weights
    test_blend = np.zeros_like(test_list[0])
    for wi, pi in zip(best_w, test_list, strict=True):
        test_blend += float(wi) * pi
    test_pred_idx = np.argmax(test_blend, axis=1)
    test_pred_label = np.array([classes[int(i)] for i in test_pred_idx], dtype=np.int64)

    test_df = pd.read_csv(args.test_csv)
    out = pd.DataFrame({"id": test_df["id"].astype(int), "Class": test_pred_label.astype(int)})
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved blended submission: {out_path}")

    if args.out_weights_json:
        w_path = Path(args.out_weights_json)
        w_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights": best_w, "oof_macro_f1": best_f1, "classes": classes}
        w_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved weights: {w_path}")


if __name__ == "__main__":
    main()


