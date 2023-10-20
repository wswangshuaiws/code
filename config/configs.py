import torch
import os


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# root
ROOT_DIR = os.path.abspath(".")

# data
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "dataset.txt")

# output
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.npz")
MODEL_MLP_PATH = os.path.join(OUTPUT_DIR, "model_MLP_%d.pkl")
MODEL_GAN_PATH = os.path.join(OUTPUT_DIR, "model_GAN_%d.pkl")
SCORE_MLP_PATH = os.path.join(OUTPUT_DIR, "score_MLP.npy")
SCORE_GAN_PATH = os.path.join(OUTPUT_DIR, "score_GAN.npy")
makedir(OUTPUT_DIR)

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model configs
CONFIG = {
    "batch_size": 5,
    "lr": 0.0005,
    "epoch": 100,
    "dropout_rate": 0.2
}
