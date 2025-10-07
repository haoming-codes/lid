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
