import os
import torch

BASE_DIR = os.path.join(os.path.expanduser("~"), "Desktop/frcnn_update")

TRAIN_DIR = os.path.join(BASE_DIR, "data/training")
VALID_DIR = os.path.join(BASE_DIR, "data/validation")
OUT_DIR = os.path.join(BASE_DIR, "models/outputs")
INFER_DIR = os.path.join(BASE_DIR, "models/inference")

# Major architecture configurations and hyper-parameters
# Batch size should be changed to a minimum value of 128 whereby not using google colab
# Learning rate is not necessary to alter due to an incorporated scheduler in the main execution loop

BATCH_SIZE = 16
WIDTH, HEIGHT = 224, 224
NUM_EPOCHS = 25
NUM_WORKERS = 2
LEARNING_RATE = 0.0001
GAMMA = 0.1
MEAN = [0.485, 0.456, 0.406]
SD = [0.229, 0.224, 0.225]

# Detection threshold representative of IoU
# Formative value was obtained via manual iteration
# 0.8 constitutes as the optimal and effective upper-bounded valye to consider

DETECTION_THRESHOLD = 0.8
FRAME_COUNT = 0
TOTAL_FPS = 0

CLASSES = ["background", "windTurbine"]
NUM_CLASSES = len(CLASSES)
VISUALISE_TRANSFORMED_IMAGES = False

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
