import json
import os
from datasets import DatasetDict, Dataset, load_dataset
import numpy as np


def load_or_create_dataset(data_dir="data", filename="phishing_url_dataset.json"):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, filename)

    if os.path.exists(file_path):
        print("Loading dataset from local storage...")
        return load_local_dataset(file_path)

    print("Creating new dataset...")
    dataset = load_dataset("pirocheto/phishing-url")
    rand_idx_train = np.random.randint(7657, size=3500)
    rand_idx_test, rand_idx_validation = list(range(0, 2000)), list(range(2000, 3500))

    x_train = dataset['train'][rand_idx_train]['url']
    y_train = [1 if x == 'legitimate' else 0 for x in dataset['train'][rand_idx_train]['status']]
    x_test = dataset['test'][rand_idx_test]['url']
    y_test = [1 if x == 'legitimate' else 0 for x in dataset['test'][rand_idx_test]['status']]
    x_validation = dataset['test'][rand_idx_validation]['url']
    y_validation = [1 if x == 'legitimate' else 0 for x in dataset['test'][rand_idx_validation]['status']]

    dataset = DatasetDict({
        'train': Dataset.from_dict({'label': y_train, 'text': x_train}),
        'test': Dataset.from_dict({'label': y_test, 'text': x_test}),
        'validation': Dataset.from_dict({'label': y_validation, 'text': x_validation})
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