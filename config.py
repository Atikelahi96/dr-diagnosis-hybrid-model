# config.py

import os

# Device
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# Paths
TRAIN_PATH = os.getenv('DR_TRAIN_PATH', 'DR_Sorted_Images_train')
VALID_PATH = os.getenv('DR_VALID_PATH', 'DR_Sorted_Images_val')
TEST_PATH  = os.getenv('DR_TEST_PATH',  'DR_Sorted_Images_test')

# Hyperparameters
NUM_CLASSES    = 5
BATCH_SIZE     = 8
EPOCHS         = 20
LEARNING_RATE  = 1e-4
MOMENTUM       = 0.9
WEIGHT_DECAY   = 1e-4
EARLY_STOPPING = 5
