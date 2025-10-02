import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import image_uris

role = sagemaker.get_execution_role()

image_uri = image_uris.retrieve(
    framework="pytorch-neuron",             # Neuron DLC family
    region="us-west-2",
    image_scope="training"                  # get the training image
)

estimator = PyTorch(
    entry_point="train_sm.py",
    source_dir="src",
    role=role,
    instance_type="ml.trn1.2xlarge",
    instance_count=1,
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.8.0-neuronx-py311-sdk2.26.0-ubuntu22.04",
    max_run=3600,  # your 1-hour cap
    # environment={"TOKENIZERS_PARALLELISM": "false"},
    hyperparameters={
        # read manifests and audio directly from S3
        "train-manifest": "s3://us-west-2-ehmli/lid-job/manifests/data_train_0930.s3.json",
        "eval-manifest":  "s3://us-west-2-ehmli/lid-job/manifests/data_valid_0930.s3.json",
        "num-epochs": 10,
        "batch-size": 8,
        "lr": 1e-4,
        "fp16": False,
        "bf16": True,
        "audio-col": "wav",
        "label-col": "lang",
        # not used for s3://, but kept for local fallbacks
        "path-root": "",
    },
    dependencies=["requirements.txt"],
)

# No TrainingInput channels needed; the script pulls from S3 itself
estimator.fit()  # inputs=None by default
