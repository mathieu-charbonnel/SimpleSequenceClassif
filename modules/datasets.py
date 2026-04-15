from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from modules.constants import AMINO_ACID_ALPHABET, MAX_SEQUENCE_SIZE
from modules.device import device


class SeqCatDataset(Dataset):
    """Dataset for paired sequence/categorical classification using one-hot encoding."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence: str,
        seq_encoder: Callable[[str, list[str], int], torch.Tensor],
        categories: list[str],
        cat_encoders: list[Any],
    ) -> None:
        self.dataframe = dataframe
        self._sequence = sequence
        self._seq_encoder = seq_encoder
        self._categories = categories
        self._cat_encoders = cat_encoders

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, target) where features is the concatenation of
        the one-hot encoded sequence and all one-hot encoded categorical columns,
        and target is the binary hit label."""
        sequence_enc = self._seq_encoder(
            self.dataframe[self._sequence].values[idx],
            AMINO_ACID_ALPHABET,
            MAX_SEQUENCE_SIZE,
        )
        encodings = [sequence_enc]

        for category, encoder in zip(self._categories, self._cat_encoders):
            encoded_cat = torch.Tensor(
                encoder.transform(
                    np.array(self.dataframe[category].values[idx]).reshape(1, -1)
                )
            ).to(device)
            encodings.append(encoded_cat)

        features = torch.hstack(tuple(encodings))
        target = self.dataframe["hit"].values[idx].reshape(1, 1)
        target = torch.Tensor(target).float()

        return features, target


class SeqCatBalancedDataset(Dataset):
    """Dataset for paired sequence/categorical classification with class balancing.

    Uses a tokenizer (e.g. BERT) for sequence encoding and computes class weights
    for upsampling to handle imbalanced datasets.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence: str,
        tokenizer: Any,
        categories: list[str],
        cat_encoders: list[Any],
    ) -> None:
        self.dataframe = dataframe
        self._sequence = sequence
        self._tokenizer = tokenizer
        self._categories = categories
        self._cat_encoders = cat_encoders

        positive_class_count = (self.dataframe["hit"] == 1).sum()
        negative_class_count = (self.dataframe["hit"] == 0).sum()
        total_samples = len(self.dataframe)
        positive_weight = (1.0 / positive_class_count) * (total_samples / 2.0)
        negative_weight = (1.0 / negative_class_count) * (total_samples / 2.0)
        self.class_weights = torch.Tensor([negative_weight, positive_weight])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (input_ids, attention_mask, categorical_features, target).

        The sequence is tokenized (e.g. by BERT), categorical columns are
        one-hot encoded and concatenated, and target is the binary hit label.
        """
        sequence_values = self.dataframe[self._sequence].values[idx]
        tokens = self._tokenizer(
            sequence_values,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQUENCE_SIZE,
        )
        sequence_enc = tokens["input_ids"].to(device)
        masks = tokens["attention_mask"].to(device)

        cat_encodings: list[torch.Tensor] = []
        for category, encoder in zip(self._categories, self._cat_encoders):
            encoded_cat = torch.Tensor(
                encoder.transform(
                    np.array(self.dataframe[category].values[idx]).reshape(1, -1)
                )
            ).to(device)
            cat_encodings.append(encoded_cat)

        cat_features = torch.hstack(tuple(cat_encodings))
        target = self.dataframe["hit"].values[idx].reshape(1, 1)
        target = torch.Tensor(target).float().to(device)

        return sequence_enc, masks, cat_features, target

    def get_weights(self) -> torch.Tensor:
        return self.class_weights


def compute_sample_weights(dataset: SeqCatBalancedDataset) -> list[float]:
    """Generate per-sample weights from dataset class weights for use with a WeightedRandomSampler."""
    cls_weights = dataset.get_weights()
    return [
        cls_weights[0].item() if hit == 0 else cls_weights[1].item()
        for hit in dataset.dataframe["hit"]
    ]


sample_weights = compute_sample_weights
