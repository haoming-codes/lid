"""Script to fine-tune facebook/mms-lid-126 for 3-class language identification.

This script expects JSON manifest(s) with entries that map utterance IDs to
metadata dictionaries. Each metadata dictionary must contain a `wav` key that
points to a local audio file on disk and a `lang` key with one of the language
labels ("en", "zh", or "other"). An optional `length` field is ignored by the
training pipeline but can remain in the manifest. Example manifest:

```
{
  "utt_7fa3a1d7ca9c": {
    "wav": "/path/to/audio.wav",
    "lang": "zh",
    "length": 2.85
  }
}
```

The script can be pointed at separate training and evaluation manifests or can
perform an automatic train/validation split from a single manifest.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from datasets import Audio, ClassLabel, Dataset, DatasetDict
import evaluate
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LANG_LABELS = ["en", "zh", "other"]


@dataclass
class FinetuneConfig:
    """Container for configuration options."""

    train_manifest: str
    eval_manifest: Optional[str]
    output_dir: str
    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    warmup_steps: int
    gradient_accumulation_steps: int
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: Optional[int]
    save_total_limit: int
    seed: int
    fp16: bool
    freeze_feature_extractor: bool
    validation_split: float
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    dataloader_num_workers: int
    preprocessing_num_workers: int


def parse_args() -> FinetuneConfig:
    parser = argparse.ArgumentParser(description="Fine-tune facebook/mms-lid-126")
    parser.add_argument(
        "--train-manifest",
        required=True,
        help="Path to the training manifest JSON file.",
    )
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help=(
            "Optional path to the evaluation manifest JSON file. If omitted and "
            "--validation-split > 0, a split will be created from the training manifest."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./mms-lid-finetuned",
        help="Directory where checkpoints and logs will be stored.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluate every N steps. Defaults to `save_steps` when not provided.",
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training.")
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Freeze the convolutional feature extractor of the backbone.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of the training manifest reserved for validation when --eval-manifest is not set.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on number of training samples for debugging.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on number of evaluation samples for debugging.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of worker processes for PyTorch DataLoader instances.",
    )
    parser.add_argument(
        "--preprocessing-num-workers",
        type=int,
        default=4,
        help="Number of worker processes to use during dataset preprocessing steps.",
    )

    args = parser.parse_args()
    return FinetuneConfig(**vars(args))


def read_manifest(path: str) -> List[Dict[str, str]]:
    """Load manifest JSON file and convert to a list of training examples."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        examples = data
    elif isinstance(data, dict):
        examples = []
        for utt_id, meta in data.items():
            if not isinstance(meta, dict):
                raise ValueError(f"Entry for {utt_id} must be a dict, got {type(meta)}")
            example = {
                "id": utt_id,
                "audio": meta.get("wav"),
                "lang": meta.get("lang"),
            }
            if not example["audio"] or not example["lang"]:
                raise ValueError(f"Entry {utt_id} must contain 'wav' and 'lang' keys.")
            examples.append(example)
    else:
        raise ValueError("Manifest must be either a list or a dict of utterance metadata.")

    for example in examples:
        if example["lang"] not in LANG_LABELS:
            raise ValueError(
                f"Found unsupported language label '{example['lang']}'. Supported labels: {LANG_LABELS}."
            )
        if not os.path.exists(example["audio"]):
            raise FileNotFoundError(f"Audio file not found: {example['audio']}")
    return examples


def build_dataset_dict(config: FinetuneConfig, feature_extractor) -> DatasetDict:
    train_examples = read_manifest(config.train_manifest)
    train_dataset = Dataset.from_list(train_examples)

    # Determine evaluation dataset
    if config.eval_manifest:
        eval_examples = read_manifest(config.eval_manifest)
        eval_dataset = Dataset.from_list(eval_examples)
    elif config.validation_split > 0.0:
        split = train_dataset.train_test_split(test_size=config.validation_split, seed=config.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        eval_dataset = None

    datasets = DatasetDict({"train": train_dataset})
    if eval_dataset is not None:
        datasets["validation"] = eval_dataset

    # Ensure consistent sampling rate and label encoding
    sampling_rate = feature_extractor.sampling_rate
    audio_feature = Audio(sampling_rate=sampling_rate)
    for split in datasets:
        datasets[split] = datasets[split].cast_column("audio", audio_feature)

    class_label = ClassLabel(names=LANG_LABELS)

    def encode_label(batch):
        batch["label"] = class_label.str2int(batch["lang"])
        return batch

    label_num_proc = config.preprocessing_num_workers if config.preprocessing_num_workers > 1 else None
    datasets = datasets.map(encode_label, num_proc=label_num_proc)

    def preprocess_batch(batch):
        audio_arrays = [record["array"] for record in batch["audio"]]
        inputs = feature_extractor(audio_arrays, sampling_rate=sampling_rate)
        batch["input_values"] = inputs["input_values"]
        if "attention_mask" in inputs:
            batch["attention_mask"] = inputs["attention_mask"]
        return batch

    for split in datasets:
        remove_cols = [col for col in datasets[split].column_names if col not in {"label"}]
        num_proc = config.preprocessing_num_workers if config.preprocessing_num_workers > 1 else None
        datasets[split] = datasets[split].map(
            preprocess_batch,
            batched=True,
            remove_columns=remove_cols,
            num_proc=num_proc,
        )
        datasets[split].set_format(type="torch")

    if config.max_train_samples is not None:
        datasets["train"] = datasets["train"].select(range(min(config.max_train_samples, len(datasets["train"]))))
    if "validation" in datasets and config.max_eval_samples is not None:
        datasets["validation"] = datasets["validation"].select(
            range(min(config.max_eval_samples, len(datasets["validation"])))
        )

    return datasets


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )


def main():
    config = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    set_seed(config.seed)

    model_name = "facebook/mms-lid-126"
    logger.info("Loading feature extractor %s", model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    logger.info("Loading datasets")
    datasets = build_dataset_dict(config, feature_extractor)

    id2label = {i: label for i, label in enumerate(LANG_LABELS)}
    label2id = {label: i for i, label in id2label.items()}

    logger.info("Loading model %s", model_name)
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=len(LANG_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    if config.freeze_feature_extractor:
        logger.info("Freezing feature extractor")
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        elif hasattr(model, "feature_extractor"):
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
        else:
            logger.warning("Model does not expose a known feature extractor; skipping freeze.")

    data_collator = DataCollatorWithPadding(feature_extractor, padding=True)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(
                predictions=preds,
                references=labels,
                average="macro",
            )["f1"],
        }
        return metrics

    eval_strategy = "steps" if "validation" in datasets else "no"
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        evaluation_strategy=eval_strategy,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
    eval_steps=(
        config.eval_steps
        if (config.eval_steps is not None and eval_strategy != "no")
        else (config.save_steps if eval_strategy != "no" else None)
    ),
        save_total_limit=config.save_total_limit,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        load_best_model_at_end=eval_strategy != "no",
        metric_for_best_model="f1_macro" if eval_strategy != "no" else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "validation" in datasets else None,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving final model to %s", config.output_dir)
    trainer.save_model(config.output_dir)
    feature_extractor.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
