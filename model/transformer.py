from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # DistilGPT-2 uses GPT-2's vocab size
    n_layer: int = 6         # DistilGPT-2 has 6 layers
    n_head: int = 12        # DistilGPT-2 has 12 attention heads
    n_embd: int = 768       # DistilGPT-2 embedding size
    dropout: float = 0.1     # DistilGPT-2 default dropout
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token
        print(f"Loaded DistilGPT-2 with {self.get_num_params() / 1e6:.2f}M parameters")

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.model.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.model.transformer.wpe.parameters())
        return n_params

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        outputs = self.model(input_ids=idx, labels=targets)
        logits = outputs.logits
        loss = outputs.loss if targets is not None else None
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        assert model_type == "distilgpt2", "Only distilgpt2 is supported for this project"
        override_args = override_args or {}
        
        config_args = {
            "distilgpt2": dict(n_layer=6, n_head=12, n_embd=768, vocab_size=50257, block_size=1024, bias=True)
        }[model_type]
        
        if "dropout" in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        
        config = GPTConfig(**config_args)
        model = cls(config)
        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str):
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            outputs = self.model(input_ids=idx_cond)
            logits = outputs.logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    def save_pretrained(self, save_directory: str):
        """Save the model and tokenizer to the specified directory."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def crop_block_size(self, block_size: int):
        """Adjust block size if needed."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size