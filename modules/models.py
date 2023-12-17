import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering
tiny_bert = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
MAX_SIZE = 15

# Define a simple neural network model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Define a model with two input features: sequence and categorical
class TinyBERTClassifier(nn.Module):
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