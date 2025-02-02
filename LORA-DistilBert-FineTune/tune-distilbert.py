import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import evaluate
from data_prepare import load_or_create_imdb_subset
from constants import *
import os


# Tokenization function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)


# Initialize model and tokenizer
def initialize_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=NUM_LABELS, id2label=LABEL_MAP, label2id={v: k for k, v in LABEL_MAP.items()})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
    tokenizer.truncation_side = "left"
    return model, tokenizer


# Evaluate function
def compute_metrics(p):
    accuracy = evaluate.load("accuracy")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}


# Fine-tune using LoRA
def apply_lora(model):
    peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['q_lin'])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# Train the model
def train_model(model, tokenizer, dataset):
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=MODEL_CHECKPOINT + "-lora-text-classification",
        learning_rate=TRAINING_ARGS["learning_rate"],
        per_device_train_batch_size=TRAINING_ARGS["batch_size"],
        per_device_eval_batch_size=TRAINING_ARGS["batch_size"],
        num_train_epochs=TRAINING_ARGS["num_epochs"],
        weight_decay=TRAINING_ARGS["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


# Predict function
def predict(model, tokenizer, texts):
    print("Model Predictions:")
    print("-----------------")
    for text in texts:
        inputs = tokenizer.encode(text, return_tensors="pt")
        logit = model(inputs).logits
        predictions = torch.argmax(logit, dim=1).item()
        print(f"{text} - {LABEL_MAP[predictions]}")


def save_model(model, tokenizer, save_directory="saved_model"):
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved in {save_directory}")


def load_model(save_directory="saved_model"):
    if os.path.exists(save_directory):
        print("Loading saved model...")
        model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        return model, tokenizer
    return None, None


# Main execution
def main():
    dataset = load_or_create_imdb_subset()
    model, tokenizer = initialize_model()
    predict(model, tokenizer, ["A cinematic masterpiece!", "Total waste of time.", "An emotional rollercoaster.",
                               "Mediocre at best.", "A must-watch for everyone.", "Lacked depth and originality.",
                               "Exceeded all my expectations!"])

    model, tokenizer = load_model()

    if model is None or tokenizer is None:
        model = apply_lora(model)
        train_model(model, tokenizer, dataset)
        save_model(model, tokenizer)

    predict(model, tokenizer, ["A cinematic masterpiece!", "Total waste of time.", "An emotional rollercoaster.",
                               "Mediocre at best.", "A must-watch for everyone.", "Lacked depth and originality.",
                               "Exceeded all my expectations!"])


if __name__ == "__main__":
    main()
