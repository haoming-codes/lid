"""Launch a SageMaker training job for finetune_mms_lid.py."""
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import image_uris

role = sagemaker.get_execution_role()

# Resolve the latest PyTorch DLC for training on GPU instances.
image_uri = image_uris.retrieve(
    framework="pytorch",
    version="2.0.1",
    py_version="py310",
    instance_type="ml.g5.2xlarge",
    image_scope="training",
    region=sagemaker.Session().boto_region_name,
)

train_manifest_uri = "s3://us-west-2-ehmli/lid-job/manifests/data_train_0930.jsonl"
validation_manifest_uri = "s3://us-west-2-ehmli/lid-job/manifests/data_valid_0930.jsonl"

estimator = PyTorch(
    entry_point="finetune_mms_lid.py",
    source_dir="src",
    role=role,
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    framework_version="2.0.1",
    py_version="py310",
    image_uri=image_uri,
    volume_size=200,  # GB; ensure there is space for audio + checkpoints
    max_run=12 * 3600,
    hyperparameters={
        # The manifests can be read directly from S3 by `datasets.load_dataset`.
        "train-manifest": train_manifest_uri,
        "eval-manifest": validation_manifest_uri,
        "output-dir": "/opt/ml/model",
        "num-train-epochs": 5,
        "learning-rate": 2e-5,
        "per-device-train-batch-size": 8,
        "per-device-eval-batch-size": 8,
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

estimator.fit()
