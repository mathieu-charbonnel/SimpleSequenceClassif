import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

from modules.device import device


def categories_fit_one_hot(
    df: pd.DataFrame,
    categories: list[str],
) -> tuple[np.ndarray, list[OneHotEncoder]]:
    """Fit one-hot encoders for categorical columns. Returns encoded features and fitted encoders."""
    full_encoding = np.empty((df.shape[0], 0))
    encoders: list[OneHotEncoder] = []

    for category in categories:
        cat_values = df[category].values
        cat_encoder = OneHotEncoder(
            sparse_output=False,
            categories="auto",
            handle_unknown="ignore",
        )
        encoded_cat = cat_encoder.fit_transform(cat_values.reshape(-1, 1))
        full_encoding = np.hstack((full_encoding, encoded_cat))
        encoders.append(cat_encoder)

    return full_encoding, encoders


def categories_transform_one_hot(
    df: pd.DataFrame,
    categories: list[str],
    encoders: list[OneHotEncoder],
) -> np.ndarray:
    """Transform categorical columns using pre-fitted one-hot encoders."""
    full_encoding = np.empty((df.shape[0], 0))

    for category, encoder in zip(categories, encoders):
        cat_values = df[category].values
        encoded_cat = encoder.transform(cat_values.reshape(-1, 1))
        full_encoding = np.hstack((full_encoding, encoded_cat))

    return full_encoding


def seq_one_hot(
    sequence: str,
    alphabet: list[str],
    max_size: int,
) -> list[list[int]]:
    """One-hot encode a sequence string, padded to max_size."""
    padded = f"{sequence:<{max_size}}"
    return [
        [1 if char == letter else 0 for char in alphabet]
        for letter in padded
    ]


def seq_pipeline(
    sequence: str,
    alphabet: list[str],
    max_size: int,
) -> torch.Tensor:
    """One-hot encode a sequence and return as a flattened tensor on the active device."""
    token = seq_one_hot(sequence, alphabet, max_size)
    tensor = torch.Tensor(token).reshape((1, -1))
    return tensor.to(device)


def df_seq_onehot_encode(
    df: pd.DataFrame,
    sequence_column: str,
    alphabet: list[str],
    max_size: int,
) -> np.ndarray:
    """One-hot encode a sequence column in a DataFrame. Returns a flattened 2D array."""
    sequences = df[sequence_column].values
    encoded = np.array(
        [seq_one_hot(seq, alphabet, max_size) for seq in sequences]
    )
    return encoded.reshape((encoded.shape[0], -1))
