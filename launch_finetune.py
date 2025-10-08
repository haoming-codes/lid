"""Launch a SageMaker training job for finetune_mms_lid.py."""
import os
from urllib.parse import urlparse

import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

role = sagemaker.get_execution_role()

# Resolve the latest PyTorch DLC for training on GPU instances.
image_uri = image_uris.retrieve(
    framework="pytorch",
    version="2.0.1",
    py_version="py310",
    instance_type="ml.g5.4xlarge",
    image_scope="training",
    region=sagemaker.Session().boto_region_name,
)
image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04"

train_manifest_uri = "s3://us-west-2-ehmli/lid-job/manifests/data_train_0930.s3.jsonl"
validation_manifest_uri = "s3://us-west-2-ehmli/lid-job/manifests/data_valid_0930.s3.jsonl"


def _local_manifest_path(channel_name: str, manifest_uri: str) -> str:
    """Return the expected local path for a manifest inside the container."""

    manifest_filename = os.path.basename(urlparse(manifest_uri).path)
    return os.path.join("/opt/ml/input/data", channel_name, manifest_filename)


train_channel = TrainingInput(
    s3_data=train_manifest_uri,
    s3_data_type="ManifestFile",
    input_mode="File",
)
validation_channel = TrainingInput(
    s3_data=validation_manifest_uri,
    s3_data_type="ManifestFile",
    input_mode="File",
)

estimator = PyTorch(
    entry_point="finetune_mms_lid.py",
    source_dir="src",
    role=role,
    instance_type="ml.g5.4xlarge",
    instance_count=1,
    # framework_version="2.0.1",
    # py_version="py310",
    image_uri=image_uri,
    # volume_size=200,  # GB; ensure there is space for audio + checkpoints
    max_run=12 * 3600,
    hyperparameters={
        # When using File mode SageMaker downloads the manifest + payload locally.
        "train-manifest": _local_manifest_path("train", train_manifest_uri),
        "eval-manifest": _local_manifest_path("validation", validation_manifest_uri),
        "output-dir": "/opt/ml/model",
        "num-train-epochs": 5,
        "learning-rate": 2e-5,
        "per-device-train-batch-size": 8,
        "per-device-eval-batch-size": 2,
        "gradient-accumulation-steps": 2,
        "warmup-steps": 500,
        "weight-decay": 0.01,
        "logging-steps": 100,
        "save-steps": 500,
        "eval-steps": 500,
        "save-total-limit": 2,
        "fp16": True,
        "freeze-feature-extractor": True,
        "initialize-other-from-average": True,
        "dataloader-num-workers": 4,
        "preprocessing-num-workers": 4,
    },
    dependencies=["requirements.txt"],
)

estimator.fit({"train": train_channel, "validation": validation_channel})
