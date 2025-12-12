from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import inspect
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src = (here.parent.parent / "src").resolve()
    sys.path.insert(0, str(src))


_ensure_src_on_path()

from social.preprocess import preprocess_social_df  # noqa: E402
from social.text_cleaning import TextNormConfig  # noqa: E402
from train_social_tfidf import build_text_feature  # noqa: E402


class TextClsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
        )
        if self.labels is not None:
            enc["labels"] = int(self.labels[idx])
        return enc


@dataclass
class LabelMap:
    label_to_id: dict[int, int]
    id_to_label: dict[int, int]


def build_label_map(y: np.ndarray) -> LabelMap:
    uniq = sorted({int(v) for v in y.tolist()})
    label_to_id = {lbl: i for i, lbl in enumerate(uniq)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    return LabelMap(label_to_id=label_to_id, id_to_label=id_to_label)


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": float(f1_score(labels, preds, average="macro"))}


def make_training_args(**kwargs) -> TrainingArguments:
    """
    Compatibility shim across Transformers versions.
    Newer versions may rename `evaluation_strategy` -> `eval_strategy`.
    We map/strip unsupported keys automatically.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    # Handle rename: evaluation_strategy -> eval_strategy
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    # Filter unsupported keys (keeps script running even if HF changes)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**filtered)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="xlm-roberta-base")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--fold_ensemble", action="store_true", help="Train K folds and average probabilities for test.")
    ap.add_argument("--save_test_proba", action="store_true", help="Save test probabilities to .npy for blending.")
    ap.add_argument("--save_oof_proba", action="store_true", help="Save OOF probabilities to .npy for blend-weight tuning.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # Use the same normalization + multi-view feature text that worked for TF-IDF.
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

    X = build_text_feature(train_df).astype(str).tolist()
    y_raw = train_df["Class"].astype(int).values
    X_test = build_text_feature(test_df).astype(str).tolist()

    lm = build_label_map(y_raw)
    y = np.array([lm.label_to_id[int(v)] for v in y_raw], dtype=np.int64)

    (out_dir / "label_map.json").write_text(
        json.dumps({"label_to_id": lm.label_to_id, "id_to_label": lm.id_to_label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    num_labels = len(lm.label_to_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: device=cpu. Transformer training will be extremely slow.\n"
            "- In Colab: Runtime -> Change runtime type -> GPU, then restart runtime.\n"
            "- Also avoid `pip install torch` (it may install CPU-only torch)."
        )
    if device != "cuda" and args.fp16:
        print("NOTE: fp16 requested but CUDA is not available; disabling fp16.")
        args.fp16 = False
    print(f"device={device} model={args.model_name} num_labels={num_labels}")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    test_proba_sum = np.zeros((len(X_test), num_labels), dtype=np.float64)
    oof_pred = np.zeros_like(y)
    oof_proba = np.zeros((len(y), num_labels), dtype=np.float64) if args.save_oof_proba else None
    fold_scores: list[float] = []

    # If not ensembling folds, just use fold 1 split for validation then refit full data.
    folds_to_train = list(skf.split(np.zeros(len(y)), y)) if args.fold_ensemble else [next(skf.split(np.zeros(len(y)), y))]

    for fold_idx, (tr_idx, va_idx) in enumerate(folds_to_train, start=1):
        print(f"\n== Fold {fold_idx}/{len(folds_to_train)} ==")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            id2label={i: str(lm.id_to_label[i]) for i in range(num_labels)},
            label2id={str(lm.id_to_label[i]): i for i in range(num_labels)},
        ).to(device)

        train_ds = TextClsDataset([X[i] for i in tr_idx], [int(y[i]) for i in tr_idx], tokenizer, args.max_len)
        val_ds = TextClsDataset([X[i] for i in va_idx], [int(y[i]) for i in va_idx], tokenizer, args.max_len)
        test_ds = TextClsDataset(X_test, None, tokenizer, args.max_len)

        run_name = f"social_tr_{Path(args.model_name).name}_fold{fold_idx}"
        training_args = make_training_args(
            output_dir=str(out_dir / run_name),
            report_to="none",
            num_train_epochs=float(args.epochs),
            learning_rate=float(args.lr),
            per_device_train_batch_size=int(args.batch_size),
            per_device_eval_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum),
            weight_decay=float(args.weight_decay),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=50,
            fp16=bool(args.fp16),
            seed=int(args.seed),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics_fn,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        print("eval:", eval_metrics)

        # OOF preds for this fold
        val_logits = trainer.predict(val_ds).predictions
        val_pred = np.argmax(val_logits, axis=1)
        oof_pred[va_idx] = val_pred
        if oof_proba is not None:
            oof_proba[va_idx] = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        fold_f1 = float(f1_score(y[va_idx], val_pred, average="macro"))
        fold_scores.append(fold_f1)

        # Test proba for ensemble
        test_logits = trainer.predict(test_ds).predictions
        test_proba = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        test_proba_sum += test_proba

        # free memory
        del trainer
        del model
        torch.cuda.empty_cache()

    oof_f1 = float(f1_score(y, oof_pred, average="macro"))
    print(f"\nOOF macro_f1={oof_f1:.5f} mean={np.mean(fold_scores):.5f} std={np.std(fold_scores):.5f}")

    # If we didn't train all folds, refit on full data for final predictions
    if not args.fold_ensemble:
        print("\nRefitting on full data for final test prediction...")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels).to(device)
        full_ds = TextClsDataset(X, [int(v) for v in y.tolist()], tokenizer, args.max_len)
        test_ds = TextClsDataset(X_test, None, tokenizer, args.max_len)

        training_args = make_training_args(
            output_dir=str(out_dir / f"social_tr_{Path(args.model_name).name}_full"),
            report_to="none",
            num_train_epochs=float(args.epochs),
            learning_rate=float(args.lr),
            per_device_train_batch_size=int(args.batch_size),
            per_device_eval_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum),
            weight_decay=float(args.weight_decay),
            evaluation_strategy="no",
            save_strategy="no",
            logging_steps=50,
            fp16=bool(args.fp16),
            seed=int(args.seed),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=full_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        trainer.train()
        test_logits = trainer.predict(test_ds).predictions
        test_proba_sum = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()

    test_pred_id = np.argmax(test_proba_sum / (args.n_splits if args.fold_ensemble else 1), axis=1)
    test_pred_label = np.array([lm.id_to_label[int(i)] for i in test_pred_id], dtype=np.int64)

    sub = pd.DataFrame({"id": test_df["id"].astype(int), "Class": test_pred_label.astype(int)})
    out_path = out_dir / "submission_social_transformer.csv"
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved submission: {out_path}")

    if args.save_test_proba:
        proba = (test_proba_sum / (args.n_splits if args.fold_ensemble else 1)).astype(np.float32)
        npy_path = out_dir / "submission_social_transformer_proba.npy"
        meta_path = out_dir / "submission_social_transformer_proba_meta.json"
        np.save(npy_path, proba)
        meta = {"classes": [int(lm.id_to_label[i]) for i in range(num_labels)], "source": "train_social_transformer.py"}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved test proba: {npy_path} + {meta_path}")

    if args.save_oof_proba:
        assert oof_proba is not None
        npy_path = out_dir / "submission_social_transformer_oof_proba.npy"
        meta_path = out_dir / "submission_social_transformer_oof_proba_meta.json"
        np.save(npy_path, oof_proba.astype(np.float32))
        meta = {"classes": [int(lm.id_to_label[i]) for i in range(num_labels)], "source": "train_social_transformer.py"}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved OOF proba: {npy_path} + {meta_path}")


if __name__ == "__main__":
    main()


