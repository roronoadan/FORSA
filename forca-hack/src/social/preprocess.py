from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .text_cleaning import (
    TextNormConfig,
    has_arabic,
    has_latin,
    keep_only_arabic,
    keep_only_latin,
    normalize_text,
)


@dataclass(frozen=True)
class SocialPaths:
    train_csv: Path
    test_csv: Path


def load_social_data(paths: SocialPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(paths.train_csv)
    test_df = pd.read_csv(paths.test_csv)
    return train_df, test_df


def social_sanity_report(
    train_df: pd.DataFrame,
    *,
    id_col: str = "id",
    label_col: str = "Class",
    text_col: str = "Commentaire client",
    top_n: int = 10,
) -> dict:
    """
    Lightweight checks for training data. Returns a dict so notebooks can pretty-print it.
    """
    report: dict = {}
    report["n_rows"] = int(train_df.shape[0])
    report["n_cols"] = int(train_df.shape[1])
    report["columns"] = list(train_df.columns)

    if id_col in train_df.columns:
        report["n_unique_id"] = int(train_df[id_col].nunique(dropna=False))
        report["n_duplicate_id"] = int(train_df.shape[0] - train_df[id_col].nunique(dropna=False))

    if label_col in train_df.columns:
        vc = train_df[label_col].value_counts(dropna=False)
        report["label_counts"] = vc.to_dict()
        report["n_labels"] = int(vc.shape[0])

    if text_col in train_df.columns:
        s = train_df[text_col].astype(str)
        report["n_empty_text"] = int((s.str.strip() == "").sum())
        report["n_missing_text"] = int(train_df[text_col].isna().sum())

        # duplicate / near-duplicate quick proxy (exact duplicates after stripping)
        stripped = s.str.replace(r"\s+", " ", regex=True).str.strip()
        report["n_exact_duplicate_text"] = int(stripped.duplicated().sum())
        report["top_duplicate_text_samples"] = (
            stripped.value_counts().head(top_n).to_dict()
            if report["n_exact_duplicate_text"] > 0
            else {}
        )

    report["missing_by_col"] = train_df.isna().sum().to_dict()
    return report


def preprocess_social_df(
    df: pd.DataFrame,
    text_col: str = "Commentaire client",
    *,
    norm_cfg: TextNormConfig | None = None,
) -> pd.DataFrame:
    """
    Adds a cleaned text column `text_clean`.
    Does NOT drop any columns (safe for debugging).
    """
    out = df.copy()
    if text_col not in out.columns:
        raise KeyError(f"Expected text column '{text_col}' in df. Got columns: {list(out.columns)}")

    raw = out[text_col].fillna("").astype(str)
    out["text_clean"] = raw.map(lambda s: normalize_text(s, norm_cfg))
    out["text_len"] = out["text_clean"].map(len)
    # Multi-view text: helps with code-switching (Darija Arabic script + French Latin script).
    out["text_ar"] = out["text_clean"].map(keep_only_arabic)
    out["text_lat"] = out["text_clean"].map(keep_only_latin)
    # FIX: Use text_clean consistently (not raw) to avoid train/test mismatch
    out["has_ar"] = out["text_clean"].map(has_arabic).astype(int)
    out["has_lat"] = out["text_clean"].map(has_latin).astype(int)
    # Lightweight signals (as tokens later): punctuation & digits are often class-correlated
    out["n_exclam"] = raw.str.count("!")
    out["n_question"] = raw.str.count(r"\?")
    out["n_digits"] = raw.str.count(r"\d")
    return out


def run_social_preprocess(
    train_csv: str | Path,
    test_csv: str | Path,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Convenience function for notebooks/scripts.
    Saves processed train/test as CSV.
    """
    paths = SocialPaths(train_csv=Path(train_csv), test_csv=Path(test_csv))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_social_data(paths)
    train_proc = preprocess_social_df(train_df)
    test_proc = preprocess_social_df(test_df)

    train_out = out_dir / "social_train_processed.csv"
    test_out = out_dir / "social_test_processed.csv"
    train_proc.to_csv(train_out, index=False, encoding="utf-8")
    test_proc.to_csv(test_out, index=False, encoding="utf-8")

    return train_out, test_out


