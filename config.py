# config.py

CONFIG = {
    "image_size": 224,
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 1e-4,
    "num_classes": 5,
    "model_name": "resnet50",
    "device": "cuda",
    "train_csv": "path/to/train.csv",
    "val_csv": "path/to/val.csv",
    "image_root": "path/to/images/",
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints"
}