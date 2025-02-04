from data_prepare import load_or_create_dataset
from constants import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
import os
import torch


# Initialize model and tokenizer
def initialize_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=NUM_LABELS,
                                                               id2label=LABEL_MAP,
                                                               label2id={v: k for k, v in LABEL_MAP.items()})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)
    tokenizer.truncation_side = "left"
    return model, tokenizer


# freeze base model parameters
def freeze_layers(model):
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False

    # unfreeze base model pooling layers to tune last 4 layers
    for name, param in model.base_model.named_parameters():
        if "pooler" in name:
            param.requires_grad = True
    return model


# Tokenization function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)


# Evaluate function
def compute_metrics(p):
    accuracy = evaluate.load("accuracy")
    predictions, labels = p
    predicted_classes = np.argmax(predictions, axis=1)
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)
    return {"Accuracy": acc}


# Train the model
def train_model(model, tokenizer, tokenized_dataset, data_collator):
    training_args = TrainingArguments(
        output_dir="bert-phishing-classifier_teacher",
        learning_rate=TRAINING_ARGS["learning_rate"],
        per_device_train_batch_size=TRAINING_ARGS["batch_size"],
        per_device_eval_batch_size=TRAINING_ARGS["batch_size"],
        num_train_epochs=TRAINING_ARGS["num_epochs"],
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


# Validate model
def validation(trainer, tokenized_dataset):
    predictions = trainer.predict(tokenized_dataset["validation"])
    logit = predictions.predictions
    labels = predictions.label_ids
    metrics = compute_metrics((logit, labels))
    print(metrics)


# Save model
def save_model(model, tokenizer, save_directory="saved_model"):
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved in {save_directory}")


# Load model
def load_model(save_directory="saved_model"):
    if os.path.exists(save_directory):
        print("Loading saved model...")
        model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        return model, tokenizer
    return None, None


# Predict function
def predict(model, tokenizer, texts):
    print("Model Predictions:")
    print("-----------------")
    for text in texts:
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
        logit = model(inputs).logits
        predictions = torch.argmax(logit, dim=1).item()
        print(f"{text} - {LABEL_MAP[predictions]}")


# Main execution
def main():
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        dataset = load_or_create_dataset()
        model, tokenizer = initialize_model()
        model = freeze_layers(model)
        tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = train_model(model, tokenizer, tokenized_dataset, data_collator)
        save_model(model, tokenizer)
        validation(trainer, tokenized_dataset)
        predict(model, tokenizer, ["https://magalu-crediarioluiza.com/Produto_20203/produto.php?sku=1962067",
                                   "http://keramikadecor.com.ua/bdfg/excelzz/bizmail.php?email={{emailb64}}"])
    else:
        predict(model, tokenizer, ["https://magalu-crediarioluiza.com/Produto_20203/produto.php?sku=1962067",
                                   "http://keramikadecor.com.ua/bdfg/excelzz/bizmail.php?email={{emailb64}}"])


if __name__ == "__main__":
    main()