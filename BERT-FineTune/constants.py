# Constants
MODEL_CHECKPOINT = "google-bert/bert-base-uncased"
LABEL_MAP = {1: "legitimate", 0: "phishing"}
NUM_LABELS = len(LABEL_MAP)
TRAINING_ARGS = {
    "learning_rate": 1e-3,
    "batch_size": 4,
    "num_epochs": 10,
    "weight_decay": 0.01,
}
