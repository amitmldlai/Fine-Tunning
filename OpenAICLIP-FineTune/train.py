import os
import json
from datasets import DatasetDict, Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import TripletEvaluator
from PIL import Image
import requests
from constants import *


def load_local_dataset(file_path):
    with open(file_path, "r") as f:
        dataset_dict = json.load(f)
    return DatasetDict({split: Dataset.from_dict(data) for split, data in dataset_dict.items()})


def process_dataset(data):
    image_list = [Image.open(requests.get(url, stream=True).raw) for url in data["thumbnail_url"]]
    return {
        "anchor": image_list,
        "positive": data["title"],
        "negative": data["title_neg"]
    }


def initialize_model(training=False):
    model = SentenceTransformer(model_name)
    if training:
        trainable_layers_list = ['projection']
        for name, param in model.named_parameters():
            param.requires_grad = False
            if any(layer in name for layer in trainable_layers_list):
                param.requires_grad = True
        return model
    return model


def evaluator(model, dataset, data_type):
    triplet_eval = TripletEvaluator(
        anchors=dataset[data_type]["anchor"],
        positives=dataset[data_type]["positive"],
        negatives=dataset[data_type]["negative"],
        name="clip-validation",
    )
    print("Accuracy", triplet_eval(model))
    return triplet_eval


def loss_function(model):
    loss = MultipleNegativesRankingLoss(model)
    return loss


def train_embedding_model(model, dataset, loss, evaluator):  # Fine-Tuning `projection` layers of model
    train_args = SentenceTransformerTrainingArguments(
        output_dir=f"models/clip_fine_tune_embeddings",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        eval_steps=1,
        logging_steps=1,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    return trainer


# def model_inference(model, dataset):
#     query_embedding = model.encode(["Your text goes here"])
#     jd_embeddings = model.encode(dataset["test"]["title_pos"])
#     similarities = model.similarity(query_embedding, jd_embeddings)
#     print(similarities.shape)
#     similar_sorted = np.argsort(similarities.numpy(), axis=1)
#     print(dataset["test"]["title_pos"][int(similar_sorted[0][-1])])


if __name__ == "__main__":
    dataset = load_local_dataset("data/you_tube_data.json")
    columns_to_remove = [col for col in dataset['train'].column_names if col not in ['anchor', 'positive', 'negative']]
    dataset = dataset.map(process_dataset, batched=True, remove_columns=columns_to_remove)
    print(dataset['train'][0])

    embed_model = initialize_model()
    train_embed_model = initialize_model(training=True)

    if os.path.exists("models/final_model"):
        print('-' * 50)
        print("Inferencing on Pre-trained Model")
        triplet_eval = evaluator(embed_model, dataset, "test")
        # model_inference(embed_model, dataset)
        print('-' * 50)
        print("Inferencing on Tuned Model")
        tuned_model = SentenceTransformer("models/final_model")  # Load the saved model
        triplet_eval = evaluator(tuned_model, dataset, "test")
        # model_inference(tuned_model, dataset)
        print('-' * 50)
    else:
        os.makedirs("models", exist_ok=True)
        triplet_eval = evaluator(train_embed_model, dataset, "valid")  # Set up evaluator
        loss = loss_function(train_embed_model)  # Set the loss function
        trainer = train_embedding_model(train_embed_model, dataset, loss, triplet_eval)  # Train the model
        triplet_eval = evaluator(train_embed_model, dataset, "test")  # Re-evaluate the model on the test set
        train_embed_model.save("models/final_model")  # Save the trained model for further inference
        print('-' * 50)
        print("Inferencing on Pre-trained Model")
        triplet_eval = evaluator(embed_model, dataset, "test")
        # model_inference(embed_model, dataset)
        print('-' * 50)
        print("Inferencing on Tuned Model")
        tuned_model = SentenceTransformer("models/final_model")  # Load the saved model
        # model_inference(tuned_model, dataset)
        print('-' * 50)
