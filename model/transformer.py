from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

class GPTConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 50257)
        self.block_size = kwargs.get("block_size", 128)
        self.n_layer = kwargs.get("n_layer", 6)
        self.n_head = kwargs.get("n_head", 12)
        self.n_embd = kwargs.get("n_embd", 768)
        self.dropout = kwargs.get("dropout", 0.1)

class GPT:
    def __init__(self, config):
        self.config = config
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(cls, pretrained_model_name, override_args=None):
        config = GPTConfig(**(override_args or {}))
        instance = cls(config)
        instance.model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        instance.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        return instance

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load_pretrained(self, save_directory):
        self.model = GPT2LMHeadModel.from_pretrained(save_directory)
        self.tokenizer = GPT2Tokenizer.from_pretrained(save_directory)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self