import torch
import torch.nn as nn

from modules.constants import MAX_SEQUENCE_SIZE


def load_tiny_bert() -> nn.Module:
    from transformers import AutoModelForQuestionAnswering
    return AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")


class SimpleClassifier(nn.Module):
    """Simple feedforward neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: concatenated features. Output: probability in [0, 1]."""
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class TinyBERTClassifier(nn.Module):
    """Combines TinyBERT sequence processing with categorical features for binary classification."""

    def __init__(
        self,
        hidden_dim: int,
        num_categories: int,
        tinybert: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tinybert = tinybert if tinybert is not None else load_tiny_bert()
        self.fc_sequence = nn.Linear(MAX_SEQUENCE_SIZE, hidden_dim)
        self.fc_categorical = nn.Linear(num_categories, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_combined = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_categorical: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Processes sequence through TinyBERT and categorical features
        through a linear layer, concatenates both, and outputs a probability in [0, 1]."""
        outputs_sequence = self.tinybert(
            input_ids=input_ids.reshape(-1, MAX_SEQUENCE_SIZE),
            attention_mask=attention_mask.reshape(-1, MAX_SEQUENCE_SIZE),
        )
        end_logits_sequence = outputs_sequence.end_logits
        x_sequence = self.fc_sequence(end_logits_sequence)

        x_categorical = self.fc_categorical(input_categorical)

        x_sequence = x_sequence.reshape(-1, 1, self.hidden_dim)
        combined_features = torch.cat([x_sequence, x_categorical], dim=2)
        x = self.relu(combined_features)
        x = self.fc_combined(x)
        x = self.sigmoid(x)
        return x
