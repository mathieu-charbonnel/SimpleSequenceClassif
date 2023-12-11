import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")

MAX_SIZE = 15
ALPHABET = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', \
            'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


class SeqCatDataset(Dataset):
    def __init__(self,
                 dataframe,
                 sequence,
                 seq_encoder,
                 categories,
                 cat_encoders):
        self.dataframe = dataframe
        self._sequence = sequence
        self._seq_encoder = seq_encoder
        self._categories = categories
        self._cat_encoders = cat_encoders

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        # Get encoded features
        sequence_enc = self._seq_encoder(
            self.dataframe[self._sequence].values[idx], ALPHABET, MAX_SIZE)
        encodings = [sequence_enc]

        for category, encoder in zip(self._categories, self._cat_encoders):
            encoded_cat = torch.Tensor(
                encoder.transform(
                (np.array(self.dataframe[category].values[idx])).reshape(1, -1))
            ).to(device)

            encodings.append(encoded_cat)

        # Concatenate the one-hot encoded features
        features = torch.hstack(tuple(encodings))
        target = self.dataframe['hit'].values[idx].reshape(1, 1)

        # Convert target to PyTorch tensor
        target = torch.Tensor(target).float()

        return features, target
    

class SeqCatBalancedDataset(SeqCatDataset):
    def __init__(self,
                 dataframe,
                 sequence,
                 seq_encoder,
                 categories,
                 cat_encoders):
        super().__init__(dataframe, 
                         sequence, 
                         seq_encoder,
                         categories,
                         cat_encoders)
        # Calculate class weights for upsampling
        positive_class_count = (self.dataframe['hit'] == 1).sum()
        negative_class_count = (self.dataframe['hit'] == 0).sum()
        total_samples = len(self.dataframe)
        positive_weight = (1.0 / positive_class_count) * (total_samples / 2.0)
        negative_weight = (1.0 / negative_class_count) * (total_samples / 2.0)
        self.class_weights = torch.Tensor([negative_weight, positive_weight])
        self.sequences = self.tokenize_sequences(sequence)

    def tokenize_sequences(self, sequence):              
        sequence_values = list(self.dataframe[sequence].values)
        tokenized_training_data = tokenizer(
            sequence_values,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_SIZE)
        return (tokenized_training_data["input_ids"].to(device))    

    def __getitem__(self, idx):

        # Get encoded features
        sequence_enc = self.sequences[idx]

        cat_encodings = []
        for category, encoder in zip(self._categories, self._cat_encoders):
            encoded_cat = torch.Tensor(
                encoder.transform(
                (np.array(self.dataframe[category].values[idx])).reshape(1, -1))
            ).to(device)

            cat_encodings.append(encoded_cat)

        # Concatenate the one-hot encoded features
        cat_features = torch.hstack(tuple(cat_encodings))
        target = self.dataframe['hit'].values[idx].reshape(1, 1)

        # Convert target to PyTorch tensor
        target = torch.Tensor(target).float()

        return sequence_enc, cat_features, target

    def get_weights(self):
      return self.class_weights

def sample_weights(dataset):
    cls_weights = dataset.get_weights()
    sample_weights = [
        cls_weights[0] if hit == 0 else cls_weights[1]
        for hit in dataset.dataframe['hit']
    ]
    return sample_weights