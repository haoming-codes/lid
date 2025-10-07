# src/train_sm.py
import fsspec
import os, sys, subprocess, argparse, json, random, numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# # ---- optional: install extra deps shipped with the job ----
# REQ = "/opt/ml/code/requirements.txt"
# if os.path.exists(REQ):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQ])

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import Counter

# =========================
# 0) Config (defaults)
# =========================
MODEL_NAME = "facebook/mms-lid-126"
TARGET_SR = 16000
CROP_SECONDS_TRAIN = 2.0
CROP_SECONDS_EVAL  = 2.0
UNFREEZE_AT_EPOCH  = 1
UNFREEZE_LAST_K    = 2
SEED = 42

THREE_WAY_LABELS = ["cmn", "eng", "other"]
label2id = {lab: i for i, lab in enumerate(THREE_WAY_LABELS)}
id2label = {i: lab for lab, i in label2id.items()}

# =========================
# 1) Utils
# =========================
class PrettyEvalLogger(TrainerCallback):
    def __init__(self, id2label_map): self.id2label_map = id2label_map

    @staticmethod
    def _gather(prefix: str, logs: dict):
        out = {}
        for k, v in logs.items():
            if k.startswith(prefix):
                name = k.split("/", 1)[1] if "/" in k else k[len(prefix):]
                out[name] = float(v)
        return out

    @staticmethod
    def _display_label(lbl: str) -> str:
        if lbl == "cmn": return "zh"
        if lbl == "eng": return "en"
        return lbl

    def _ordered_labels(self) -> List[str]:
        ordered = [self.id2label_map[i] for i in sorted(self.id2label_map)]
        return [self._display_label(lbl) for lbl in ordered]

    def _format_confusion(self, logs: dict) -> str:
        labels = self._ordered_labels()
        def _blank_row():
            return {pred: 0 for pred in labels}
        matrix = {lab: _blank_row() for lab in labels}
        for k, v in logs.items():
            if not k.startswith("eval_cm/"): continue
            key = k.split("/", 1)[1]
            if "->" not in key: continue
            tgt, pred = key.split("->", 1)
            if tgt not in matrix: matrix[tgt] = _blank_row()
            if pred not in matrix[tgt]:
                matrix[tgt][pred] = 0
            matrix[tgt][pred] = int(v)
        header = "pred> " + " ".join(f"{p:>6}" for p in labels)
        rows = []
        for tgt in labels:
            row = " ".join(f"{matrix.get(tgt, {}).get(pred, 0):>6d}" for pred in labels)
            rows.append(f"true={tgt:>3} {row}")
        return "\n".join([header] + rows)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "eval_loss" not in logs: return
        loss = float(logs.get("eval_loss", float("nan")))
        acc = float(logs.get("eval_accuracy", float("nan")))
        per_class_acc  = self._gather("eval_acc/", logs)
        per_class_prec = self._gather("eval_precision/", logs)
        per_class_rec  = self._gather("eval_recall/", logs)
        labels = self._ordered_labels()
        per_class_lines = []
        for lbl in labels:
            acc_v = per_class_acc.get(lbl, float("nan"))
            prec_v = per_class_prec.get(lbl, float("nan"))
            rec_v = per_class_rec.get(lbl, float("nan"))
            per_class_lines.append(f"  {lbl:>3}: acc={acc_v:6.3f}, prec={prec_v:6.3f}, rec={rec_v:6.3f}")
        confusion_str = self._format_confusion(logs)
        print(
            "\n".join(
                [
                    f"Eval step @ {state.global_step}: loss={loss:.4f}, acc={acc:.4f}",
                    "Per-class metrics:",
                    *per_class_lines,
                    "Confusion matrix:",
                    confusion_str,
                ]
            ),
            flush=True,
        )

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def crop_or_pad(wav: torch.Tensor, num_samples: int, random_crop: bool) -> torch.Tensor:
    L = wav.numel()
    if L == num_samples: return wav
    if L > num_samples:
        start = np.random.randint(0, L - num_samples + 1) if random_crop else max(0, (L - num_samples) // 2)
        return wav[start:start+num_samples]
    pad = num_samples - L
    left, right = pad // 2, pad - (pad // 2)
    return torch.nn.functional.pad(wav, (left, right))

def _auto_cols(cols: List[str]) -> Tuple[str, str]:
    audio_candidates = ["local_filename","audio","audio_path","path","wav","audio_filepath","file","fname"]
    label_candidates = ["label","lang","language","target","y"]
    a = next((c for c in audio_candidates if c in cols), None)
    y = next((c for c in label_candidates if c in cols), None)
    if a is None or y is None:
        raise ValueError(f"Could not infer audio/label columns from {cols}. Pass --audio-col and --label-col.")
    return a, y

def normalize_lang(s: Any) -> str:
    l = str(s).strip().lower()
    if l in {"zh","zho","cmn","mandarin","zh-cn","zh-hans","zh-hant","zh-tw","zh-hk","chinese"}: return "cmn"
    if l in {"en","eng","english","en-us","en-gb"}: return "eng"
    return "other"

# =========================
# 2) Manifest -> Dataset
# =========================
def is_s3_uri(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")

def load_manifest_to_dataset(manifest_path: str, audio_col: Optional[str], label_col: Optional[str], path_root: Optional[str]) -> Dataset:
    # If it's JSON/JSONL on S3, HF Datasets can read it directly as long as s3fs is installed.
    # (falls back to fsspec for non-standard shapes)
    try:
        ds = load_dataset("json", data_files=manifest_path, split="train")  # supports s3:// via fsspec/s3fs
    except Exception:
        # manual read (works for dict/list/JSONL)
        open_fn = (lambda p: fsspec.open(p, "rb").open()) if is_s3_uri(manifest_path) else (lambda p: open(p, "rb"))
        with open_fn(manifest_path) as f:
            raw = f.read()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            # JSONL fallback
            data = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        rows = [dict(v, utt_id=k) for k, v in data.items()] if isinstance(data, dict) else data
        ds = Dataset.from_list(rows)

    a_col, y_col = (audio_col, label_col) if (audio_col and label_col) else _auto_cols(ds.column_names)

    # only join local relative paths; leave s3:// alone
    if path_root:
        def _fix(ex):
            p = ex[a_col]
            if p and (not is_s3_uri(p)) and (not os.path.isabs(p)):
                ex[a_col] = os.path.normpath(os.path.join(path_root, p))
            return ex
        ds = ds.map(_fix)

    ds = ds.cast_column(a_col, Audio(sampling_rate=TARGET_SR))

    def _has_audio_and_label(ex):
        audio = ex.get(a_col)
        if isinstance(audio, dict):
            has_audio = bool(audio.get("path")) or (audio.get("array") is not None)
        else:
            has_audio = bool(audio)
        return has_audio and bool(ex.get(y_col))

    ds = ds.filter(_has_audio_and_label)
    ds = ds.map(lambda ex: {**ex, y_col: normalize_lang(ex[y_col])})
    keep = [c for c in [a_col, y_col, "utt_id"] if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds._hf_audio_col = a_col; ds._hf_label_col = y_col
    ds.set_format(type=None)
    return ds

# =========================
# 3) Preprocessing
# =========================
def make_preprocess_fn(processor, is_train: bool, audio_col: str, label_col: str):
    target_len = int((CROP_SECONDS_TRAIN if is_train else CROP_SECONDS_EVAL) * TARGET_SR)
    def _fn(example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example[audio_col]
        wav = torch.from_numpy(audio["array"]).float()
        wav = crop_or_pad(wav, target_len, random_crop=is_train)
        inputs = processor(wav.numpy(), sampling_rate=TARGET_SR, return_attention_mask=False)
        lab = example[label_col]
        return {"input_values": inputs["input_values"][0], "labels": int(label2id[lab])}
    return _fn

@dataclass
class DataCollatorAudioClassification:
    processor: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        input_values = [f["input_values"] for f in features]
        batch = self.processor.pad({"input_values": input_values}, return_tensors="pt")
        batch["labels"] = labels
        return batch

# =========================
# 4) Metrics
# =========================
def compute_metrics(eval_pred):
    num_labels = len(id2label)
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    overall_acc = float((preds == labels).mean().item())
    cm = confusion_matrix(labels, preds, labels=list(range(num_labels)))
    denom = np.where(cm.sum(axis=1) == 0, 1, cm.sum(axis=1))
    per_class_acc = cm.diagonal() / denom
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_labels)), average=None, zero_division=0
    )
    def disp(lbl: str) -> str: return "zh" if lbl == "cmn" else ("en" if lbl == "eng" else lbl)
    metrics = {"accuracy": overall_acc}
    for i in range(num_labels):
        name = disp(id2label.get(i, f"class_{i}"))
        metrics[f"acc/{name}"]       = float(per_class_acc[i])
        metrics[f"precision/{name}"] = float(prec[i])
        metrics[f"recall/{name}"]    = float(rec[i])
    labels_disp = [disp(id2label[i]) for i in range(num_labels)]
    for ti, tname in enumerate(labels_disp):
        for pj, pname in enumerate(labels_disp):
            metrics[f"cm/{tname}->{pname}"] = int(cm[ti, pj])
    return metrics

# =========================
# 5) Unfreeze callback
# =========================
class UnfreezeLastKLayers(TrainerCallback):
    def __init__(self, k: int, at_epoch: int): self.k = k; self.at_epoch = at_epoch; self.done = False
    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.done or self.at_epoch is None: return
        if state.epoch is not None and state.epoch >= self.at_epoch:
            model = kwargs["model"]
            try: layers = model.wav2vec2.encoder.layers
            except AttributeError:
                print("[WARN] Could not find wav2vec2.encoder.layers to unfreeze."); self.done = True; return
            for p in model.wav2vec2.feature_extractor.parameters(): p.requires_grad = False
            for layer in layers[-self.k:]:
                for p in layer.parameters(): p.requires_grad = True
            print(f"[INFO] Unfroze last {self.k} encoder blocks at epoch {state.epoch:.2f}")
            self.done = True

# =========================
# 6) Smart init for 3-way head
# =========================
@torch.no_grad()
def smart_init_three_way_head(model_3: Wav2Vec2ForSequenceClassification,
                              model_126: Wav2Vec2ForSequenceClassification):
    old_head = getattr(model_126, "classifier", None); old_W = old_head.weight.data.clone(); old_b = old_head.bias.data.clone()
    id2lab_126 = {int(k): v for k, v in model_126.config.id2label.items()}
    def find_idx(cands): 
        cset = {c.lower() for c in cands}
        for i, lab in id2lab_126.items():
            if str(lab).lower() in cset: return i
        return None
    idx_eng = find_idx(["eng","en","english"]); idx_cmn = find_idx(["cmn","zho","zh","chinese"])
    new_head = getattr(model_3, "classifier", None); new_W = new_head.weight.data; new_b = new_head.bias.data
    mask = torch.ones_like(old_b, dtype=torch.bool)
    if idx_eng is not None: mask[idx_eng] = False
    if idx_cmn is not None: mask[idx_cmn] = False
    mean_W = old_W[mask].mean(dim=0); mean_b = old_b[mask].mean()
    def set_row(new_label, src_idx, fbW, fbB):
        j = model_3.config.label2id[new_label]
        if src_idx is not None: new_W[j] = old_W[src_idx]; new_b[j] = old_b[src_idx]
        else: new_W[j] = fbW; new_b[j] = fbB
    set_row("cmn", idx_cmn, mean_W, mean_b); set_row("eng", idx_eng, mean_W, mean_b); set_row("other", None, mean_W, mean_b)

def class_balanced_weights(counts: dict, beta: float = 0.9999):
    weights = {}
    for lab, n in counts.items(): weights[lab] = (1.0 - beta) / (1.0 - beta**max(1, n))
    return torch.tensor([weights["cmn"], weights["eng"], weights["other"]], dtype=torch.float)

class ImbTrainer(Trainer):
    def __init__(self, *args, class_weights=None, focal_gamma=None, **kwargs):
        super().__init__(*args, **kwargs); self.class_weights = class_weights; self.focal_gamma = focal_gamma
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs["labels"]; model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs); logits = outputs.logits
        cw = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=cw, reduction="none")
        if self.focal_gamma is not None:
            with torch.no_grad(): pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = ((1 - pt) ** self.focal_gamma * ce).mean()
        else: loss = ce.mean()
        return (loss, outputs) if return_outputs else loss

# =========================
# 7) Train/Eval entrypoint
# =========================
def train_lid_with_hf_from_manifests(
    train_manifest: str,
    eval_manifest: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
    fp16: bool,
    bf16: bool,
    audio_col: Optional[str],
    label_col: Optional[str],
    path_root: Optional[str],
):
    set_seed(SEED)

    train_ds = load_manifest_to_dataset(train_manifest, audio_col, label_col, path_root)
    eval_ds  = load_manifest_to_dataset(eval_manifest , audio_col, label_col, path_root)
    a_col = getattr(train_ds, "_hf_audio_col"); y_col = getattr(train_ds, "_hf_label_col")

    processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    base_126  = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(THREE_WAY_LABELS),
        label2id=label2id, id2label=id2label, problem_type="single_label_classification",
        ignore_mismatched_sizes=True
    )
    if hasattr(model, "freeze_feature_encoder"): model.freeze_feature_encoder()
    else:
        for p in model.wav2vec2.feature_extractor.parameters(): p.requires_grad = False
    smart_init_three_way_head(model, base_126)

    train_ds_proc = train_ds.shuffle(seed=SEED).map(
        make_preprocess_fn(processor, True,  a_col, y_col),
        remove_columns=train_ds.column_names,
        num_proc=max(1, min(os.cpu_count(), 16)),        # parallel workers
        batched=False,             # process mini-batches per worker
        # batch_size=32,    # larger = fewer Python calls; watch RAM
    )
    eval_ds_proc  = eval_ds.map(
        make_preprocess_fn(processor, False, a_col, y_col),
        remove_columns=eval_ds.column_names,
        num_proc=max(1, min(os.cpu_count(), 16)),        # parallel workers
        batched=False,             # process mini-batches per worker
        # batch_size=32,    # larger = fewer Python calls; watch RAM
    )
    collator = DataCollatorAudioClassification(processor)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(2, batch_size // 2),
        num_train_epochs=num_epochs,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=False,
        report_to="none",
        logging_strategy="steps", logging_steps=500,
        eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )

    callbacks = []
    if UNFREEZE_AT_EPOCH is not None and UNFREEZE_LAST_K > 0:
        callbacks.append(UnfreezeLastKLayers(UNFREEZE_LAST_K, UNFREEZE_AT_EPOCH))
    callbacks.append(PrettyEvalLogger(id2label))

    train_counts = Counter(train_ds[y_col]); print("Train label counts:", dict(train_counts))
    w = class_balanced_weights(train_counts, beta=0.9999)
    focal_gamma = None

    trainer = ImbTrainer(
        model=model,
        args=args,
        train_dataset=train_ds_proc,
        eval_dataset=eval_ds_proc,
        processing_class=processor,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=w,
        focal_gamma=focal_gamma,
    )
    trainer.train()
    print("Eval:", trainer.evaluate())
    return trainer, processor, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # file paths (relative to training channel by default)
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--eval-manifest",  type=str, required=True)
    # hparams
    parser.add_argument("--output-dir",   type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--num-epochs",   type=int, default=10)
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--fp16",         type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--bf16",         type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--audio-col",    type=str, default=None)
    parser.add_argument("--label-col",    type=str, default=None)
    parser.add_argument("--path-root",    type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    args = parser.parse_args()

    trainer, processor, model = train_lid_with_hf_from_manifests(
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        audio_col=args.audio_col,
        label_col=args.label_col,
        path_root=args.path_root,
    )
    # Save HF-formatted artifacts to SM_MODEL_DIR so SageMaker uploads them
    save_dir = os.environ.get("SM_MODEL_DIR", args.output_dir)
    processor.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
