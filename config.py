import torch

# -- Data Configuration --
# TODO: Change this to the actual path of your dataset
DATA_DIR = "data"
NUM_KEYPOINTS = 10
IMG_SIZE = 224

# -- Model Configuration --
MODEL_BACKBONE = 'mobilenetv4_conv_small.e2400_r224_in1k'

# -- Training Configuration --
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

# For resuming training, set to "checkpoints/last_model.pth" for example
RESUME_CHECKPOINT = None

# -- Loss Function Configuration --
WING_LOSS_W = 10.0
WING_LOSS_EPSILON = 2.0
VISIBILITY_LOSS_LAMBDA = 1.0

# -- Checkpoint and Export Configuration --
CHECKPOINT_SAVE_DIR = "checkpoints"
BEST_MODEL_NAME = "best_model.pth"
LAST_MODEL_NAME = "last_model.pth"
EXPORTED_MODEL_NAME = "model_traced.pt"
