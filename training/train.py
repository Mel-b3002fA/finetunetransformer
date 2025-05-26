import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    # Explicitly set device (CPU for Intel Mac, as CUDA/MPS is unavailable or unreliable)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    
    # Set pad_token to eos_token to avoid padding issues
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Apply dropout override
    model.config.dropout = 0.1  # Set dropout directly in model config
    
    # Load a sample dataset (wikitext for demonstration; replace with your dataset)
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    try:
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        return
    
    # Remove unnecessary columns and set format
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=True,  # Explicitly disable CUDA for macOS compatibility
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else tokenized_dataset["test"],
    )
    
    # Train the model
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main()