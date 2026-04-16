# Goal: Single source of truth for all hyperparameters
# config.py
PROJECT_ROOT_DIR = (
    "/content/drive/MyDrive/ai-projects/2026-03-transfer-learning-resnet-50"
)
LOG_DIR = PROJECT_ROOT_DIR + "/logs"

DATASET_NAME = "COVID-QU-Ex Dataset"
DATASET_ROOT_DIR = "/content/drive/MyDrive/ai-datasets"
COVIDQU_ZIP_PATH = DATASET_ROOT_DIR + "/covidqu.zip"
COVIDQU_PATH = DATASET_ROOT_DIR + "/covidqu"

IMG_SIZE = (224, 224)
# Add seed for reproducibility:
SEED = 42

# actual run
BATCH_SIZE = 128
LEARNING_RATE = 0.004

# Add normalization constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EPOCHS = 15

DEVICE = "cuda"
