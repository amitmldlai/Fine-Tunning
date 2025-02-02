import os
import json
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset


def load_or_create_imdb_subset(sample_size=1000, data_dir="data", filename="imdb_subset.json"):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)

    if os.path.exists(file_path):
        print("Loading dataset from local storage...")
        return load_local_dataset(file_path)

    print("Creating new dataset...")
    imdb_dataset = load_dataset("imdb")
    rand_idx = np.random.randint(24999, size=sample_size)

    x_train = imdb_dataset['train'][rand_idx]['text']
    y_train = imdb_dataset['train'][rand_idx]['label']
    x_test = imdb_dataset['test'][rand_idx]['text']
    y_test = imdb_dataset['test'][rand_idx]['label']

    dataset = DatasetDict({
        'train': Dataset.from_dict({'label': y_train, 'text': x_train}),
        'validation': Dataset.from_dict({'label': y_test, 'text': x_test})
    })

    save_dataset(dataset, file_path)
    return dataset


def save_dataset(dataset, file_path):
    dataset_dict = {split: dataset[split].to_dict() for split in dataset}
    with open(file_path, "w") as f:
        json.dump(dataset_dict, f)


def load_local_dataset(file_path):
    with open(file_path, "r") as f:
        dataset_dict = json.load(f)
    return DatasetDict({split: Dataset.from_dict(data) for split, data in dataset_dict.items()})
