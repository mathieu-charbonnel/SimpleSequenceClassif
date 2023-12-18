import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering
tiny_bert = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
MAX_SIZE = 15

class SimpleClassifier(nn.Module):
    """
    A simple feedforward neural network for binary classification.

    Args:
    - input_dim (int): Dimensionality of the input features.
    - hidden_dim (int): Dimensionality of the hidden layer.

    Attributes:
    - fc1 (nn.Linear): The first fully connected layer.
    - fc2 (nn.Linear): The second fully connected layer.
    - sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
    - forward(x): Forward pass of the classifier.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
        - x (torch.Tensor): Input features.

        Returns:
        - torch.Tensor: The output of the classifier after the sigmoid activation.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class TinyBERTClassifier(nn.Module):
    """
    A classifier model that combines TinyBERT-based sequence processing
    with categorical feature processing for binary classification.

    Args:
    - hidden_dim (int): Dimensionality of the hidden layers.
    - num_categories (int): Number of categories in the categorical input.

    Attributes:
    - hidden_dim (int): Dimensionality of the hidden layers.
    - tinybert (nn.Module): TinyBERT model for sequence processing.
    - fc_sequence (nn.Linear): Fully connected layer for sequence features.
    - fc_categorical (nn.Linear): Fully connected layer for categorical features.
    - relu (nn.ReLU): Rectified Linear Unit activation function.
    - fc_combined (nn.Linear): Fully connected layer for combined features.
    - sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
    - forward(input_ids, attention_mask, input_categorical):
      Forward pass of the classifier.
    """

    def __init__(self, hidden_dim, num_categories):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tinybert = tiny_bert
        self.fc_sequence = nn.Linear(MAX_SIZE, hidden_dim)
        self.fc_categorical = nn.Linear(num_categories, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_combined = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids,
                attention_mask,
                input_categorical):
        """
        Forward pass of the classifier.

        Args:
        - input_ids (torch.Tensor): Input sequence IDs
        - attention_mask (torch.Tensor): Attention mask for sequence 
          input
        - input_categorical (torch.Tensor): Input categorical features

        Returns:
        - torch.Tensor: The output of the classifier after the sigmoid
          activation
        """
        # Process sequence input with TinyBERT
        outputs_sequence = self.tinybert(
            input_ids=input_ids.reshape(-1, 15),
            attention_mask=attention_mask.reshape(-1, 15)
        )
        end_logits_sequence = outputs_sequence.end_logits
        x_sequence = self.fc_sequence(end_logits_sequence)

        # Process categorical input
        x_categorical = self.fc_categorical(input_categorical)

        # Combine the features
        x_sequence = x_sequence.reshape(-1, 1, self.hidden_dim)
        combined_features = torch.cat([x_sequence, x_categorical], dim=2)
        x = self.relu(combined_features)
        x = self.fc_combined(x)
        x = self.sigmoid(x)
        return x
