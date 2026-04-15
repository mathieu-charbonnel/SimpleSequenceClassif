import numpy as np
import pandas as pd
import torch

from modules.preprocessing import (
    df_seq_onehot_encode,
    seq_one_hot,
    seq_pipeline,
)


class TestSeqOneHot:
    def test_basic_encoding(self) -> None:
        alphabet = ["A", "B", "C"]
        result = seq_one_hot("AB", alphabet, 3)
        assert result == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]

    def test_padding(self) -> None:
        alphabet = ["A", "B"]
        result = seq_one_hot("A", alphabet, 3)
        assert len(result) == 3
        assert result[0] == [1, 0]
        assert result[1] == [0, 0]
        assert result[2] == [0, 0]

    def test_full_sequence(self) -> None:
        alphabet = ["A", "B", "C"]
        result = seq_one_hot("ABC", alphabet, 3)
        assert result == [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

    def test_unknown_character(self) -> None:
        alphabet = ["A", "B"]
        result = seq_one_hot("X", alphabet, 2)
        assert result[0] == [0, 0]

    def test_output_dimensions(self) -> None:
        alphabet = ["A", "R", "N", "D", "C"]
        result = seq_one_hot("ARN", alphabet, 5)
        assert len(result) == 5
        assert all(len(row) == 5 for row in result)


class TestSeqPipeline:
    def test_output_is_tensor(self) -> None:
        alphabet = ["A", "B", "C"]
        result = seq_pipeline("AB", alphabet, 3)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self) -> None:
        alphabet = ["A", "B", "C"]
        result = seq_pipeline("AB", alphabet, 3)
        assert result.shape == (1, 9)

    def test_values_match_one_hot(self) -> None:
        alphabet = ["A", "B"]
        result = seq_pipeline("A", alphabet, 2)
        expected = torch.Tensor([[1, 0, 0, 0]])
        assert torch.allclose(result.cpu(), expected)


class TestDfSeqOnehotEncode:
    def test_output_shape(self) -> None:
        df = pd.DataFrame({"seq": ["AB", "CD"]})
        alphabet = ["A", "B", "C", "D"]
        result = df_seq_onehot_encode(df, "seq", alphabet, 3)
        assert result.shape == (2, 12)

    def test_output_values_are_binary(self) -> None:
        df = pd.DataFrame({"seq": ["AB"]})
        alphabet = ["A", "B"]
        result = df_seq_onehot_encode(df, "seq", alphabet, 3)
        assert set(np.unique(result)).issubset({0, 1})
