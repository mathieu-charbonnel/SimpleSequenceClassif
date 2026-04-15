import torch
import pytest

from modules.models import SimpleClassifier


class TestSimpleClassifier:
    def test_output_shape(self) -> None:
        model = SimpleClassifier(input_dim=10, hidden_dim=5)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 1)

    def test_output_range(self) -> None:
        model = SimpleClassifier(input_dim=10, hidden_dim=5)
        x = torch.randn(16, 10)
        output = model(x)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_single_sample(self) -> None:
        model = SimpleClassifier(input_dim=5, hidden_dim=3)
        x = torch.randn(1, 5)
        output = model(x)
        assert output.shape == (1, 1)

    def test_deterministic_in_eval_mode(self) -> None:
        model = SimpleClassifier(input_dim=10, hidden_dim=5)
        model.eval()
        x = torch.randn(4, 10)
        output1 = model(x)
        output2 = model(x)
        assert torch.allclose(output1, output2)

    def test_different_hidden_dims(self) -> None:
        for hidden_dim in [1, 16, 64, 256]:
            model = SimpleClassifier(input_dim=20, hidden_dim=hidden_dim)
            x = torch.randn(2, 20)
            output = model(x)
            assert output.shape == (2, 1)
