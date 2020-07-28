# file for constant

#####################
## hyperparameters ##
#####################

BASE_LR = 0.001
MOMENTUM = 0.9 
NUM_EPOCHES = 18
WEIGHT_DECAY = 0.005
TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 12
LR_DECAY_STEP_SIZE = 6
LR_DECAY_GAMMA = 0.1

NUM_CLASSES = 101 
NUM_MOTION_LABEL = 14

CLIP_LENGTH = 16
CROP_SIZE = 112
RESIZE_H = 171
RESIZE_W = 128

#####################
### path to data  ###
#####################

TEST_LIST = 'list/test_ucf101.list'
TRAIN_LIST = 'list/rgb_train_linux.list'
FRAMELIST_FILE = 'list/rgb_train_linux.list'
V_FLOW_LIST_FILE = 'list/v_flow_train_linux.list'
U_FLOW_LIST_FILE = 'list/u_flow_train_linux.list'
MODEL_DIR = 'models'

#####################
### path to model  ##
#####################

TRAINED_MODEL = 'c3d-finetune-2.pth-60000'
TRAIN_MODEL_NAME = 'c3d-finetune.pth'
PRETRAINED_MODEL = 'c3d-motion-0804.pth-60000'
PRETRAIN_MODEL_NAME = 'c3d-motion.pth'

