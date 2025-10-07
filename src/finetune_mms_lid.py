"""Script to fine-tune facebook/mms-lid-126 for 3-class language identification.

This script expects JSON Lines (jsonl) manifest(s) with one JSON object per
line. Each object must contain an `id` (or `utt_id`), a `wav` (or `audio`) key
that points to an audio file on disk, and a `lang` key with one of the language
labels ("en", "zh", or "other"). An optional `length` field is ignored by the
training pipeline but can remain in the manifest. Example manifest:

```
{"id": "utt_7fa3a1d7ca9c", "wav": "/path/to/audio.wav", "lang": "zh", "length": 2.85}
{"id": "utt_c80a11b7c96e", "audio": "/path/to/another.wav", "lang": "other"}
```

The script can be pointed at separate training and evaluation manifests or can
perform an automatic train/validation split from a single manifest.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from datasets import Audio, ClassLabel, Dataset, DatasetDict, load_dataset
import evaluate
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LANG_LABELS = ["en", "zh", "other"]

# ISO 639-3 codes representing the 126 language classes in the
# facebook/mms-lid-126 checkpoint. The order matches the classifier head.
MMS_LID_LANGUAGE_CODES = [
    "ara",
    "cmn",
    "eng",
    "spa",
    "fra",
    "mlg",
    "swe",
    "por",
    "vie",
    "ful",
    "sun",
    "asm",
    "ben",
    "zlm",
    "kor",
    "ind",
    "hin",
    "tuk",
    "urd",
    "aze",
    "slv",
    "mon",
    "hau",
    "tel",
    "swh",
    "bod",
    "rus",
    "tur",
    "heb",
    "mar",
    "som",
    "tgl",
    "tat",
    "tha",
    "cat",
    "ron",
    "mal",
    "bel",
    "pol",
    "yor",
    "nld",
    "bul",
    "hat",
    "afr",
    "isl",
    "amh",
    "tam",
    "hun",
    "hrv",
    "lit",
    "cym",
    "fas",
    "mkd",
    "ell",
    "bos",
    "deu",
    "sqi",
    "jav",
    "nob",
    "uzb",
    "snd",
    "lat",
    "nya",
    "grn",
    "mya",
    "orm",
    "lin",
    "hye",
    "yue",
    "pan",
    "jpn",
    "kaz",
    "npi",
    "kat",
    "guj",
    "kan",
    "tgk",
    "ukr",
    "ces",
    "lav",
    "bak",
    "khm",
    "fao",
    "glg",
    "ltz",
    "lao",
    "mlt",
    "sin",
    "sna",
    "ita",
    "srp",
    "mri",
    "nno",
    "pus",
    "eus",
    "ory",
    "lug",
    "bre",
    "luo",
    "slk",
    "fin",
    "dan",
    "yid",
    "est",
    "ceb",
    "war",
    "san",
    "kir",
    "oci",
    "wol",
    "haw",
    "kam",
    "umb",
    "xho",
    "epo",
    "zul",
    "ibo",
    "abk",
    "ckb",
    "nso",
    "gle",
    "kea",
    "ast",
    "sco",
    "glv",
    "ina",
]

# Languages that predominantly represent the "other" label in our downstream
# dataset (East and Southeast Asia, excluding China but including India).
OTHER_CLASS_LANGUAGE_CODES = [
    "asm",
    "ben",
    "hin",
    "urd",
    "tel",
    "tam",
    "mal",
    "mar",
    "pan",
    "guj",
    "kan",
    "ory",
    "san",
    "npi",
    "sin",
    "pus",
    "vie",
    "sun",
    "zlm",
    "kor",
    "ind",
    "jav",
    "tgl",
    "tha",
    "mon",
    "mya",
    "khm",
    "lao",
    "ceb",
    "war",
    "jpn",
]


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
    train_classifier_only: bool
    validation_split: float
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    dataloader_num_workers: int
    preprocessing_num_workers: int
    initialize_other_from_average: bool


def _str_to_bool(value: str | bool) -> bool:
    """Parse a string representation of truth into a boolean value."""

    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, received '{value}'.")


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
    parser.add_argument(
        "--fp16",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable FP16 training.",
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Freeze the convolutional feature extractor of the backbone.",
    )
    parser.add_argument(
        "--train-classifier-only",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Freeze all model parameters except the classification head.",
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
    parser.add_argument(
        "--initialize-other-from-average",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Initialize the 'other' classification head using the average of selected"
            " MMS LID languages (East and Southeast Asia, excluding China)."
        ),
    )

    args = parser.parse_args()
    return FinetuneConfig(**vars(args))


def _is_remote_path(path: str) -> bool:
    """Return True if *path* points to a non-local resource."""

    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.scheme != "file")


def load_manifest_dataset(path: str) -> Dataset:
    """Load and validate a jsonl manifest using the 🤗 Datasets loader."""

    try:
        dataset = load_dataset("json", data_files=path, split="train")
    except Exception as exc:  # pragma: no cover - surfaces config errors
        raise ValueError(f"Failed to load manifest {path}: {exc}") from exc

    if dataset.num_rows == 0:
        raise ValueError(f"Manifest {path} did not contain any entries.")

    if "id" not in dataset.column_names and "utt_id" in dataset.column_names:
        dataset = dataset.rename_column("utt_id", "id")

    required_columns = {"id", "lang"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest {path} is missing required column(s): {missing_list}.")

    if "audio" not in dataset.column_names:
        if "wav" in dataset.column_names:
            dataset = dataset.rename_column("wav", "audio")
        else:
            raise ValueError(
                f"Manifest {path} must include an 'audio' (or 'wav') column with audio file paths."
            )

    ids = dataset["id"]
    if any(not example_id for example_id in ids):
        raise ValueError(f"Manifest {path} contains entries with missing 'id' values.")

    langs = set(dataset["lang"])
    invalid_langs = sorted(langs - set(LANG_LABELS))
    if invalid_langs:
        raise ValueError(
            "Found unsupported language label(s) "
            f"{invalid_langs} in manifest {path}. Supported labels: {LANG_LABELS}."
        )

    audio_paths = dataset["audio"]
    if any(not audio_path for audio_path in audio_paths):
        raise ValueError(f"Manifest {path} contains entries with missing audio paths.")

    # For remote manifests (e.g., s3:// URIs) the individual audio paths are also remote and
    # will be fetched lazily by 🤗 Datasets. In that case we skip the local existence check. The
    # datasets library – backed by fsspec – will surface a clear error if the remote objects are
    # missing.
    if not _is_remote_path(path):
        missing_files = [
            audio_path
            for audio_path in audio_paths
            if not _is_remote_path(audio_path) and not os.path.exists(audio_path)
        ]
        if missing_files:
            raise FileNotFoundError(
                "Audio file(s) not found: " + ", ".join(sorted(set(missing_files)))
            )

    return dataset


def build_dataset_dict(config: FinetuneConfig, feature_extractor) -> DatasetDict:
    train_dataset = load_manifest_dataset(config.train_manifest)

    # Determine evaluation dataset
    if config.eval_manifest:
        eval_dataset = load_manifest_dataset(config.eval_manifest)
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
    # Let 🤗 Datasets handle decoding/resampling via the Audio feature. This supports local paths,
    # S3 URIs, and other fsspec-compatible locations without custom loaders.
    audio_feature = Audio(sampling_rate=sampling_rate)
    for split in datasets:
        datasets[split] = datasets[split].cast_column("audio", audio_feature)

    class_label = ClassLabel(names=LANG_LABELS)

    def encode_label(batch):
        batch["label"] = class_label.str2int(batch["lang"])
        return batch

    label_num_proc = config.preprocessing_num_workers if config.preprocessing_num_workers > 1 else None
    datasets = datasets.map(encode_label, num_proc=label_num_proc)

    logger = logging.getLogger(__name__)

    def _to_array(audio_value: dict) -> np.ndarray:
        """Convert the dataset Audio value into a float32 numpy array."""
        return np.asarray(audio_value["array"], dtype=np.float32)

    length_column = "length" if "length" in datasets["train"].column_names else None
    temp_length_column = None

    if length_column is None:
        temp_length_column = "_temp_audio_length"

        def compute_audio_length(batch):
            batch[temp_length_column] = [len(audio_value["array"]) for audio_value in batch["audio"]]
            return batch

        num_proc = config.preprocessing_num_workers if config.preprocessing_num_workers > 1 else None
        datasets["train"] = datasets["train"].map(
            compute_audio_length,
            batched=True,
            num_proc=num_proc,
        )
        length_column = temp_length_column

    train_lengths = datasets["train"][length_column]
    if not train_lengths:
        raise ValueError("Training dataset does not contain any audio examples to compute length statistics.")
    max_audio_samples = max(1, int(np.percentile(train_lengths, 95)))

    if temp_length_column is not None:
        datasets["train"] = datasets["train"].remove_columns(temp_length_column)
    logger.info(
        "Capping audio to the 95th percentile: %d samples (%.2f seconds)",
        max_audio_samples,
        max_audio_samples / float(sampling_rate),
    )

    def crop_array(array: np.ndarray) -> np.ndarray:
        if array.shape[0] <= max_audio_samples:
            return array
        start = (array.shape[0] - max_audio_samples) // 2
        end = start + max_audio_samples
        return array[start:end]

    def preprocess_batch(batch):
        cropped = [crop_array(_to_array(audio_value)) for audio_value in batch["audio"]]
        inputs = feature_extractor(
            cropped,
            sampling_rate=sampling_rate,
            max_length=max_audio_samples,
            truncation=True,
            return_attention_mask=True,
        )
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
    logger.info(
        "Initializing classification head with %d labels and ignoring mismatched head weights",
        len(LANG_LABELS),
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=len(LANG_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if config.initialize_other_from_average:
        logger.info("Initializing 'other' class weights from MMS LID language average")
        base_model = AutoModelForAudioClassification.from_pretrained(model_name)
        source_classifier = getattr(base_model, "classifier", None)
        target_classifier = getattr(model, "classifier", None)

        if source_classifier is None or target_classifier is None:
            logger.warning("Could not access classifier modules; skipping custom initialization.")
        elif not hasattr(source_classifier, "weight") or not hasattr(target_classifier, "weight"):
            logger.warning("Classifier modules do not expose weights; skipping custom initialization.")
        else:
            iso_to_index = {code: idx for idx, code in enumerate(MMS_LID_LANGUAGE_CODES)}
            missing_codes = [code for code in OTHER_CLASS_LANGUAGE_CODES if code not in iso_to_index]
            if missing_codes:
                logger.warning(
                    "Some requested languages are missing from MMS LID head (%s); skipping custom initialization.",
                    ", ".join(missing_codes),
                )
            else:
                indices = [iso_to_index[code] for code in OTHER_CLASS_LANGUAGE_CODES]
                source_weight = source_classifier.weight.detach()
                source_bias = (
                    source_classifier.bias.detach() if hasattr(source_classifier, "bias") and source_classifier.bias is not None else None
                )

                device = target_classifier.weight.device
                avg_weight = source_weight[indices].mean(dim=0).to(device)
                with torch.no_grad():
                    target_classifier.weight[label2id["other"]].copy_(avg_weight)
                    if (
                        source_bias is not None
                        and hasattr(target_classifier, "bias")
                        and target_classifier.bias is not None
                    ):
                        avg_bias = source_bias[indices].mean().to(device)
                        target_classifier.bias[label2id["other"]].copy_(avg_bias)
                logger.info(
                    "Initialized 'other' class weights using %d MMS LID languages.",
                    len(indices),
                )

        del base_model

    if config.train_classifier_only:
        classifier = getattr(model, "classifier", None)
        if classifier is None:
            logger.warning(
                "Model does not expose a 'classifier' attribute; cannot train classifier only."
            )
        else:
            logger.info("Freezing all parameters except the classifier head.")
            for param in model.parameters():
                param.requires_grad = False
            for param in classifier.parameters():
                param.requires_grad = True

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

        label_indices = list(range(len(LANG_LABELS)))
        precisions, recalls, _, _ = precision_recall_fscore_support(
            labels,
            preds,
            labels=label_indices,
            zero_division=0,
        )
        conf_matrix = confusion_matrix(labels, preds, labels=label_indices)

        for idx, label in enumerate(LANG_LABELS):
            logger.info(
                "Eval metrics for class '%s': precision=%.4f recall=%.4f",
                label,
                float(precisions[idx]),
                float(recalls[idx]),
            )
        logger.info("Eval confusion matrix:\n%s", conf_matrix)

        return metrics

    eval_strategy = "steps" if "validation" in datasets else "no"
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_strategy=eval_strategy,
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
        eval_on_start=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        processing_class=feature_extractor,
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
