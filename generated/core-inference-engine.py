"""Implementation of the core algorithm from the paper "How much do language models memorize?" (https://www.semanticscholar.org/paper/92c7e90d006c59e20f8ba093fd04189563dc474f).

The algorithm trains a causal language model on a given corpus and estimates memorization by measuring the average cross‑entropy loss on a held‑out subset of training examples (the “memorization set”). Lower loss indicates that the model has memorized those examples.

Key hyperparameters (with defaults):
- memorization_fraction (float): fraction of training examples used as the memorization set (default 0.1).
- epochs (int): number of training epochs (default 3).
- batch_size (int): mini‑batch size (default 8).
- lr (float): learning rate for AdamW optimizer (default 5e-5).
- max_seq_length (int): maximum token sequence length (default 128).
- model_name (str): pretrained model identifier (default "gpt2").
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class MemorizationDataset(Dataset):
    """Dataset that tokenizes texts and pads them to a fixed length.

    Args:
        texts (List[str]): List of raw text strings.
        tokenizer (AutoTokenizer): Tokenizer to convert texts to IDs.
        max_length (int): Maximum sequence length; longer sequences are truncated,
            shorter ones are padded.
    """

    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Tokenize all texts once
        self.encodings = tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        self.input_ids = self.encodings["input_ids"]
        self.attention_mask = self.encodings["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }


def core_inference_engine(
    train_texts: List[str],
    memorization_fraction: float = 0.1,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_seq_length: int = 128,
    model_name: str = "gpt2",
) -> float:
    """Train a causal language model on *train_texts* and return the average loss
    on a held‑out memorization subset.

    The algorithm follows the paper's core procedure:
    1. Split the training texts into a main training set and a memorization set
       of size ``memorization_fraction`` of the total size (randomly shuffled).
    2. Tokenize the texts with a pretrained tokenizer.
    3. Train the model for ``epochs`` epochs using AdamW optimizer.
    4. Evaluate the model on the memorization set (no gradient) and compute the
       average cross‑entropy loss per token.

    Args:
        train_texts: List of raw text strings used for training.
        memorization_fraction: Fraction of the data to reserve for memorization
            evaluation (default 0.1).
        epochs: Number of training epochs (default 3).
        batch_size: Mini‑batch size (default 8).
        lr: Learning rate for the optimizer (default 5e-5).
        max_seq_length: Maximum token sequence length (default 128).
        model_name: Identifier of the pretrained model to load (default "gpt2").

    Returns:
        float: Average cross‑entropy loss (per token) on the memorization set.
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()  # Ensure training mode

    # Shuffle indices and split into training vs. memorization sets
    indices = list(range(len(train_texts)))
    torch.manual_seed(42)  # For reproducibility
    torch.shuffle(indices)
    split_idx = int(len(indices) * (1 - memorization_fraction))
    train_indices = indices[:split_idx]
    mem_indices = indices[split_idx:]

    # Create datasets
    train_dataset = MemorizationDataset([train_texts[i] for i in train_indices],
                                        tokenizer, max_seq_length)
    mem_dataset = MemorizationDataset([train_texts[i] for i in mem_indices],
                                      tokenizer, max_seq_length)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    mem_loader = DataLoader(mem_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Shift logits/labels for causal language modeling
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]  # next-token prediction

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Evaluation on memorization set (no gradient)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in mem_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Sum loss over tokens in the batch
            batch_loss = loss.item() * input_ids.size(1) - input_ids[:, 0].sum().item()
            # The above correction accounts for the last token having no label
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss


if __name__ == "__main__":
    # Example usage with a tiny synthetic corpus
    example_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In computer science, AI enables machines to learn.",
        "Samsung develops cutting‑edge semiconductor technology.",
        "Language models can memorize specific training examples.",
        "Edge devices require efficient inference algorithms.",
    ]

    loss = core_inference_engine(example_texts, memorization_fraction=0.4, epochs=2)
    print(f"Average memorization loss (per token): {loss:.4f}")