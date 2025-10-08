Usage
```
python finetune_mms_lid.py
--train-manifest ../manifests/data_train_0930.json
--eval-manifest ../manifests/data_valid_0930.json
--per-device-train-batch-size 2
--per-device-eval-batch-size 2
--preprocessing-batch-size 16
--initialize-other-from-average
--fp16
```

## Using S3 manifests on Amazon SageMaker

When you launch a SageMaker training job you do not need to manually copy the
objects referenced by a manifest into the container. Configure your training
input channel to use the manifest directly and keep the default `File`
`input_mode`. SageMaker will materialize both the manifest file and every S3
object it lists inside `/opt/ml/input/data/<channel_name>/` before your script
starts, so `finetune_mms_lid.py` can open them as regular local files and
benefit from multiprocessing.

```python
from sagemaker.inputs import TrainingInput

train_input = TrainingInput(
    s3_data="s3://my-bucket/path/to/train.manifest",
    s3_data_type="ManifestFile",
    input_mode="File",  # ensures downloads to /opt/ml/input/data/<channel>
)

estimator.fit({"train": train_input})
```

If you also supply an evaluation manifest, configure a second channel (for
example `{"train": train_input, "validation": validation_input}`) and SageMaker
will download it to `/opt/ml/input/data/validation/` using the same mechanism.
