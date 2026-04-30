import torch

# -- Data Configuration --
DATA_DIR = "data"
NUM_KEYPOINTS = 10
IMG_SIZE = 384
# Keypoint index pairs that swap when the image is flipped horizontally
FLIP_PAIRS = [(0, 4), (1, 5), (8, 9)]

# -- Model Configuration --
MODEL_BACKBONE = 'mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k'

# -- Training Configuration --
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 96
EPOCHS = 75
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 1.5e-4
WARMUP_EPOCHS = 5
MIN_LR = 1.5e-6
FREEZE_BACKBONE_EPOCHS = 10
BACKBONE_LR_SCALE = 0.1

# For resuming training, set to "checkpoints/last_model.pth" for example
RESUME_CHECKPOINT = None

# -- Loss Function Configuration --
WING_LOSS_W = 10.0
WING_LOSS_EPSILON = 2.0
VISIBILITY_LOSS_LAMBDA = 1.0

EARLY_STOPPING_PATIENCE = 10

# -- Checkpoint and Export Configuration --
CHECKPOINT_SAVE_DIR = "checkpoints"
BEST_MODEL_NAME = "best_model.pth"
LAST_MODEL_NAME = "last_model.pth"
EXPORTED_MODEL_NAME = "model_traced.pt"
