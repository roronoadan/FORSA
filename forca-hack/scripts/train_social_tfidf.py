from __future__ import annotations

import argparse
import json
import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src = (here.parent.parent / "src").resolve()
    sys.path.insert(0, str(src))


_ensure_src_on_path()

from social.preprocess import preprocess_social_df, social_sanity_report  # noqa: E402
from social.text_cleaning import TextNormConfig  # noqa: E402


def build_text_feature(df: pd.DataFrame) -> pd.Series:
    # Add platform token to help the model learn platform-specific phrasing.
    if "Réseau Social" in df.columns:
        platform = df["Réseau Social"].fillna("unknown").astype(str)
    else:
        # Keep it safe if column name changes or is missing.
        platform = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")

    # Multi-view text improves code-switching robustness (Arabic vs Latin script).
    # We encode simple signals as tokens instead of numeric features so TF-IDF can use them.
    def _safe_col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    text_clean = _safe_col("text_clean")
    text_ar = _safe_col("text_ar")
    text_lat = _safe_col("text_lat")
    has_ar = _safe_col("has_ar")
    has_lat = _safe_col("has_lat")
    n_exclam = _safe_col("n_exclam")
    n_question = _safe_col("n_question")
    n_digits = _safe_col("n_digits")

    return (
        "[PLAT="
        + platform
        + "] "
        + "[HAS_AR="
        + has_ar
        + "] "
        + "[HAS_LAT="
        + has_lat
        + "] "
        + "[N_EXCL="
        + n_exclam
        + "] "
        + "[N_Q="
        + n_question
        + "] "
        + "[N_DIG="
        + n_digits
        + "] "
        + " "
        + text_clean
        + " [AR] "
        + text_ar
        + " [LAT] "
        + text_lat
    ).fillna("")


def make_vectorizer(
    kind: str,
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
) -> TfidfVectorizer:
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
    if kind == "word":
        # Use a permissive token pattern to keep Arabic words and short tokens.
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
    raise ValueError(f"Unknown vectorizer kind: {kind}")


def make_model(
    *,
    seed: int = 42,
    use_word: bool = True,
    char_max_features: int = 200_000,
    word_max_features: int = 80_000,
    char_ngram: tuple[int, int] = (3, 6),
    word_ngram: tuple[int, int] = (1, 2),
    min_df: int = 2,
    C: float = 4.0,
    clf: str = "lr",
    alpha: float = 1e-5,
    calib_cv: int = 3,
) -> Pipeline:
    if use_word:
        feats = FeatureUnion(
            [
                (
                    "char",
                    make_vectorizer(
                        "char", max_features=char_max_features, ngram_range=char_ngram, min_df=min_df
                    ),
                ),
                (
                    "word",
                    make_vectorizer(
                        "word", max_features=word_max_features, ngram_range=word_ngram, min_df=min_df
                    ),
                ),
            ]
        )
    else:
        feats = make_vectorizer("char", max_features=char_max_features, ngram_range=char_ngram, min_df=min_df)

    if clf == "lr":
        classifier = LogisticRegression(
            max_iter=4000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
            solver="saga",
            C=C,
        )
    elif clf == "sgd":
        # Fast linear model that often complements LR in ensembles.
        # log_loss => predict_proba available (needed for blending).
        classifier = SGDClassifier(
            loss="log_loss",
            alpha=float(alpha),
            penalty="l2",
            max_iter=2000,
            tol=1e-3,
            random_state=seed,
            class_weight="balanced",
        )
    elif clf == "svm":
        # Linear SVM is often very strong on TF-IDF; we calibrate to get predict_proba for blending.
        base = LinearSVC(C=float(C), class_weight="balanced", random_state=seed)
        classifier = CalibratedClassifierCV(base, cv=int(calib_cv), method="sigmoid")
    else:
        raise ValueError(f"Unknown clf: {clf} (expected 'lr' or 'sgd')")

    return Pipeline([("tfidf", feats), ("clf", classifier)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # Backward-compatible flags:
    # - default is True (recommended). Use --no_word to force char-only.
    # - --use_word remains accepted but is redundant if you don't pass --no_word.
    ap.add_argument("--use_word", action="store_true", help="(deprecated) kept for compatibility.")
    ap.add_argument("--no_word", action="store_true", help="Disable word TF-IDF (char-only).")
    ap.add_argument("--C", type=float, default=6.0)
    ap.add_argument("--search", action="store_true", help="Try a small set of strong configs and keep the best.")
    ap.add_argument(
        "--include_svm",
        action="store_true",
        help="Include calibrated LinearSVC in --search candidates (slower but can boost leaderboard).",
    )
    ap.add_argument(
        "--search_mode",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="How wide the TF-IDF config search should be (only used with --search).",
    )
    ap.add_argument("--ensemble_top_k", type=int, default=1, help="If >1 and --search, ensemble top-k configs.")
    ap.add_argument(
        "--predict_strategy",
        type=str,
        default="fold_ensemble",
        choices=["full", "fold_ensemble"],
        help="How to predict test: full=fit once on all data, fold_ensemble=avg probs across CV folds.",
    )
    ap.add_argument("--save_artifacts", action="store_true", help="Save CV artifacts (confusion matrix, per-class F1, configs).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    print("== Sanity report (train) ==")
    rep = social_sanity_report(train_df)
    print({k: rep[k] for k in ["n_rows", "n_cols", "n_labels", "n_duplicate_id"] if k in rep})

    # "Best practice" normalization config for Darija/French noisy comments.
    # Keep it explicit so we can A/B test later.
    norm_cfg = TextNormConfig(
        keep_url_token=True,
        keep_mention_token=True,
        keep_hashtag_text=True,
        keep_hashtag_token=True,
        mask_phone_numbers=True,
        reduce_elongations=True,
        lowercase_latin=True,
        deaccent_latin=False,
        keep_digits=True,
        arabizi_map_digits=True,
    )
    train_df = preprocess_social_df(train_df, norm_cfg=norm_cfg)
    test_df = preprocess_social_df(test_df, norm_cfg=norm_cfg)

    if "Class" not in train_df.columns:
        raise KeyError("Expected label column 'Class' in train.csv")
    if "id" not in train_df.columns or "id" not in test_df.columns:
        raise KeyError("Expected 'id' column in both train and test")

    X = build_text_feature(train_df)
    y = train_df["Class"].astype(int).values
    X_test = build_text_feature(test_df)
    classes = np.sort(np.unique(y))

    use_word = True
    if args.no_word:
        use_word = False
    elif args.use_word:
        use_word = True

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    def run_cv_and_maybe_predict(
        cfg: dict, *, do_predict: bool
    ) -> tuple[float, np.ndarray | None, np.ndarray]:
        oof = np.empty_like(y)
        fold_scores: list[float] = []
        test_proba_sum = np.zeros((X_test.shape[0], classes.shape[0]), dtype=np.float64) if do_predict else None

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            model = make_model(seed=args.seed + fold, **cfg)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            oof[va_idx] = pred
            score = f1_score(y[va_idx], pred, average="macro")
            fold_scores.append(float(score))
            print(f"fold {fold}: macro_f1={score:.5f}")

            if do_predict and args.predict_strategy == "fold_ensemble":
                proba = model.predict_proba(X_test)
                # Align columns defensively
                clf_classes = model.named_steps["clf"].classes_
                idx = {int(c): i for i, c in enumerate(clf_classes)}
                aligned = np.zeros_like(test_proba_sum)
                for j, c in enumerate(classes):
                    aligned[:, j] = proba[:, idx[int(c)]]
                test_proba_sum += aligned

        oof_score = float(f1_score(y, oof, average="macro"))
        print(f"OOF macro_f1={oof_score:.5f} | mean={np.mean(fold_scores):.5f} std={np.std(fold_scores):.5f}")

        if not do_predict:
            return oof_score, None, oof

        if args.predict_strategy == "fold_ensemble":
            return oof_score, test_proba_sum / args.n_splits, oof

        final_model = make_model(seed=args.seed, **cfg)
        final_model.fit(X, y)
        proba = final_model.predict_proba(X_test)
        clf_classes = final_model.named_steps["clf"].classes_
        idx = {int(c): i for i, c in enumerate(clf_classes)}
        aligned = np.zeros((X_test.shape[0], classes.shape[0]), dtype=np.float64)
        for j, c in enumerate(classes):
            aligned[:, j] = proba[:, idx[int(c)]]
        return oof_score, aligned, oof

    def save_cv_artifacts(*, cfg: dict, oof_pred: np.ndarray, tag: str) -> None:
        if not args.save_artifacts:
            return
        # Confusion matrix in fixed class order
        cm = confusion_matrix(y, oof_pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
        cm_path = out_dir / f"cv_confusion_{tag}.csv"
        cm_df.to_csv(cm_path, index=True, encoding="utf-8")

        # Per-class F1
        per_class_f1 = {int(c): float(f1_score((y == c).astype(int), (oof_pred == c).astype(int))) for c in classes}
        f1_path = out_dir / f"cv_f1_per_class_{tag}.json"
        f1_path.write_text(json.dumps(per_class_f1, ensure_ascii=False, indent=2), encoding="utf-8")

        cfg_path = out_dir / f"cv_best_config_{tag}.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.search:
        # Search space: keep "small" fast; medium/large explore more without exploding runtime.
        if args.search_mode == "small":
            candidates: list[dict] = [
                # char-only
                {"use_word": False, "char_ngram": (3, 6), "min_df": 2, "C": 6.0},
                {"use_word": False, "char_ngram": (3, 7), "min_df": 1, "C": 6.0},
                # char + word
                {"use_word": True, "char_ngram": (3, 6), "word_ngram": (1, 2), "min_df": 2, "C": 6.0},
                {"use_word": True, "char_ngram": (3, 6), "word_ngram": (1, 3), "min_df": 2, "C": 6.0},
                {"use_word": True, "char_ngram": (3, 7), "word_ngram": (1, 2), "min_df": 1, "C": 8.0},
            ]
        else:
            # Medium/Large: broaden a bit. Note: dataset is small, so this is still manageable in Colab.
            Cs = [2.0, 4.0, 6.0, 8.0, 12.0] if args.search_mode == "medium" else [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
            min_dfs = [1, 2] if args.search_mode == "medium" else [1, 2, 3]
            char_ngrams = [(3, 5), (3, 6), (3, 7), (4, 7)] if args.search_mode == "medium" else [(3, 5), (3, 6), (3, 7), (3, 8), (4, 7), (4, 8)]
            word_ngrams = [(1, 2), (1, 3)] if args.search_mode == "medium" else [(1, 2), (1, 3), (2, 4)]
            char_feats = [200_000, 300_000] if args.search_mode == "medium" else [200_000, 300_000, 400_000]
            word_feats = [80_000, 120_000] if args.search_mode == "medium" else [80_000, 120_000, 180_000]

            char_only: list[dict] = []
            char_word: list[dict] = []

            clfs = ["lr", "sgd"]
            if args.include_svm:
                clfs.append("svm")

            # Char-only configs
            for C, min_df, char_ngram, cfeat in itertools.product(Cs, min_dfs, char_ngrams, char_feats):
                for clf_name in clfs:
                    cfg = {
                        "use_word": False,
                        "char_ngram": char_ngram,
                        "min_df": int(min_df),
                        "C": float(C),
                        "char_max_features": int(cfeat),
                        "clf": clf_name,
                    }
                    if clf_name == "sgd":
                        cfg["alpha"] = 1e-5
                    if clf_name == "svm":
                        cfg["calib_cv"] = 3
                    char_only.append(cfg)

            # Char+word configs (keep word settings moderate)
            for C, min_df, char_ngram, word_ngram, cfeat, wfeat in itertools.product(
                Cs, min_dfs, char_ngrams, word_ngrams, char_feats, word_feats
            ):
                for clf_name in clfs:
                    cfg = {
                        "use_word": True,
                        "char_ngram": char_ngram,
                        "word_ngram": word_ngram,
                        "min_df": int(min_df),
                        "C": float(C),
                        "char_max_features": int(cfeat),
                        "word_max_features": int(wfeat),
                        "clf": clf_name,
                    }
                    if clf_name == "sgd":
                        cfg["alpha"] = 1e-5
                    if clf_name == "svm":
                        cfg["calib_cv"] = 3
                    char_word.append(cfg)

            # Deterministic order (helps reproducibility)
            def _key(d: dict) -> tuple:
                return (
                    d.get("clf", "lr"),
                    d.get("char_ngram"),
                    d.get("word_ngram", (0, 0)),
                    int(d.get("min_df", 2)),
                    float(d.get("C", 4.0)),
                    int(d.get("char_max_features", 0)),
                    int(d.get("word_max_features", 0)),
                    float(d.get("alpha", 0.0)),
                )

            char_only.sort(key=_key)
            char_word.sort(key=_key)

            # Safety cap to keep runtime bounded, but **preserve both groups**
            cap = 80 if args.search_mode == "medium" else 160
            half = cap // 2
            char_only_sel = char_only[:half]
            char_word_sel = char_word[:half]
            # If one side is short, refill from the other
            if len(char_only_sel) < half:
                need = half - len(char_only_sel)
                char_word_sel = char_word[: half + need]
            if len(char_word_sel) < half:
                need = half - len(char_word_sel)
                char_only_sel = char_only[: half + need]

            # Interleave so early search isn't biased
            candidates = []
            for a, b in itertools.zip_longest(char_only_sel, char_word_sel):
                if a is not None:
                    candidates.append(a)
                if b is not None:
                    candidates.append(b)

            print(
                f"Search mode={args.search_mode} candidates={len(candidates)} "
                f"(char_only={len(char_only_sel)}, char_word={len(char_word_sel)})"
            )

        scores: list[tuple[float, dict]] = []
        for i, cfg in enumerate(candidates, start=1):
            print(f"\n== Search config {i}/{len(candidates)}: {cfg} ==")
            score, _, _oof = run_cv_and_maybe_predict(cfg, do_predict=False)
            scores.append((score, cfg))

        scores.sort(key=lambda x: x[0], reverse=True)
        print("\n== Search results ==")
        for s, cfg in scores:
            print(f"macro_f1={s:.5f} cfg={cfg}")

        top_k = max(1, min(int(args.ensemble_top_k), len(scores)))
        print(f"\nUsing top_k={top_k} for test prediction.")
        test_proba_sum = np.zeros((X_test.shape[0], classes.shape[0]), dtype=np.float64)

        for rank in range(top_k):
            cfg = scores[rank][1]
            print(f"\n== Predicting with config rank {rank+1}: {cfg} ==")
            score, proba, oof_pred = run_cv_and_maybe_predict(cfg, do_predict=True)
            assert proba is not None
            test_proba_sum += proba
            save_cv_artifacts(cfg=cfg, oof_pred=oof_pred, tag=f"rank{rank+1}_macro{score:.5f}")

        test_pred = classes[np.argmax(test_proba_sum / top_k, axis=1)]
        out_name = f"submission_social_best_top{top_k}.csv"
    else:
        cfg = {"use_word": use_word, "C": float(args.C)}
        print(f"\n== Single config: {cfg} ==")
        score, proba, oof_pred = run_cv_and_maybe_predict(cfg, do_predict=True)
        assert proba is not None
        test_pred = classes[np.argmax(proba, axis=1)]
        out_name = "submission_social_tfidf.csv"
        save_cv_artifacts(cfg=cfg, oof_pred=oof_pred, tag=f"single_macro{score:.5f}")

    sub = pd.DataFrame({"id": test_df["id"].astype(int), "Class": test_pred.astype(int)})
    sub_path = out_dir / out_name
    sub.to_csv(sub_path, index=False, encoding="utf-8")
    print(f"Saved submission: {sub_path}")


if __name__ == "__main__":
    main()


