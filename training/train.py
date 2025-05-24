from model.transformer import GPT, GPTConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

def main():
    # Initialize model and config
    config = GPTConfig()
    model = GPT.from_pretrained("distilgpt2", override_args={"dropout": 0.1})

    # Load dataset
    dataset_path = "data/processed_dataset.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found. Run data/dataset.py first.")
    
    dataset = load_dataset("json", data_files=dataset_path)
    
    def tokenize_function(examples):
        encodings = model.tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./out",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        # Optimize for MacBook (CPU/MPS if available)
        no_cuda=True if not torch.cuda.is_available() else False,
        fp16=False,  # Disable FP16 for CPU/MPS compatibility
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model.model,  # Use the Hugging Face model inside GPT wrapper
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./out/finetuned_distilgpt2")
    print(f"Model saved to ./out/finetuned_distilgpt2")

if __name__ == "__main__":
    main()