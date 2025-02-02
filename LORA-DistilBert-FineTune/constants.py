# Constants
MODEL_CHECKPOINT = "distilbert-base-uncased"
LABEL_MAP = {0: "Negative", 1: "Positive"}
NUM_LABELS = len(LABEL_MAP)
TRAINING_ARGS = {
    "learning_rate": 1e-3,
    "batch_size": 4,
    "num_epochs": 10,
    "weight_decay": 0.01,
}
