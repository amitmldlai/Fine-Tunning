import os
import json
import numpy as np
from datasets import DatasetDict, Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from constants import *


def load_local_dataset(file_path):
    with open(file_path, "r") as f:
        dataset_dict = json.load(f)
    return DatasetDict({split: Dataset.from_dict(data) for split, data in dataset_dict.items()})


def initialize_model():
    model = SentenceTransformer(model_name)
    return model


def evaluator(model, dataset, data_type):
    triplet_eval = TripletEvaluator(
        anchors=dataset[data_type]["query"],
        positives=dataset[data_type]["job_description_pos"],
        negatives=dataset[data_type]["job_description_neg"],
        name="ai-job-validation",
    )
    print("Accuracy", triplet_eval(model))
    return triplet_eval


def loss_function(model):
    loss = MultipleNegativesRankingLoss(model)
    return loss


def train_embedding_model(model, dataset, loss, evaluator):    # Fine-Tuning all layers of model
    train_args = SentenceTransformerTrainingArguments(
        output_dir=f"models/distil_roberta_job_embeddings",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=100,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    return trainer


def model_inference(model, dataset):
    query_embedding = model.encode(["data scientist 6 year experience, LLMs, credit risk, content marketing"])
    jd_embeddings = model.encode(dataset["test"]["job_description_pos"])
    similarities = model.similarity(query_embedding, jd_embeddings)
    print(similarities.shape)
    similar_sorted = np.argsort(similarities.numpy(), axis=1)
    print(dataset["test"]["job_description_pos"][int(similar_sorted[0][-1])])


if __name__ == "__main__":
    dataset = load_local_dataset("data/ai-job.json")
    embed_model = initialize_model()
    if os.path.exists("models/final_model"):
        print('-' * 50)
        print("Inferencing on Pre-trained Model")
        triplet_eval = evaluator(embed_model, dataset, "test")
        model_inference(embed_model, dataset)
        print('-' * 50)
        print("Inferencing on Tuned Model")
        tuned_model = SentenceTransformer("models/final_model")  # Load the saved model
        triplet_eval = evaluator(tuned_model, dataset, "test")
        model_inference(tuned_model, dataset)
        print('-' * 50)
    else:
        os.makedirs("models", exist_ok=True)
        triplet_eval = evaluator(embed_model, dataset, "validation")  # Set up evaluator
        loss = loss_function(embed_model)  # Set the loss function
        trainer = train_embedding_model(embed_model, dataset, loss, triplet_eval)  # Train the model
        triplet_eval = evaluator(embed_model, dataset, "test")  # Re-evaluate the model on the test set
        embed_model.save("models/final_model")  # Save the trained model for further inference
        print('-' * 50)
        print("Inferencing on Pre-trained Model")
        model_inference(embed_model, dataset)
        print('-' * 50)
        print("Inferencing on Tuned Model")
        tuned_model = SentenceTransformer("models/final_model")  # Load the saved model
        model_inference(tuned_model, dataset)
        print('-' * 50)