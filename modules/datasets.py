import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import numpy as np

MAX_SIZE = 15
ALPHABET = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', \
            'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


class SeqCatDataset(Dataset):
    """
    PyTorch Dataset for a sequence categorization task.

    Args:
    - dataframe (pd.DataFrame): The input DataFrame
    - sequence (str): The column name representing the sequence data
    - seq_encoder (function): A function for encoding sequence data
    - categories (list): A list of column names representing 
      categorical features
    - cat_encoders (list): A list of encoders for categorical features

    Methods:
    - __len__(): Returns the number of samples in the dataset.
    - __getitem__(idx): Retrieves the data at the specified index.
    """

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
        """
        Returns the number of samples in the dataset.

        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves and processes the data at the specified index.

        Args:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input 
          features and target for the given index.
        """
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

class SeqCatBalancedDataset(Dataset):
    """
    PyTorch Dataset for a sequence classification task with class 
    balancing

    Args:
    - dataframe (pd.DataFrame): The input DataFrame
    - sequence (str): The column name representing the sequence data
    - tokenizer: The tokenizer for encoding sequence data.
    - categories (list): A list of column names representing 
      categorical features
    - cat_encoders (list): A list of encoders corresponding to the
      categorical features.

    Attributes:
    - class_weights (torch.Tensor): Class weights for upsampling to 
      balance classes.

    Methods:
    - __len__(): Returns the number of samples in the dataset.
    - __getitem__(idx): Retrieves and processes the data at the specified index.
    - get_weights(): Returns the class weights used for upsampling.
    """

    def __init__(self,
                 dataframe,
                 sequence,
                 tokenizer,
                 categories,
                 cat_encoders):
        self.dataframe = dataframe
        self._sequence = sequence
        self._tokenizer = tokenizer
        self._categories = categories
        self._cat_encoders = cat_encoders

        # Calculate class weights for upsampling
        positive_class_count = (self.dataframe['hit'] == 1).sum()
        negative_class_count = (self.dataframe['hit'] == 0).sum()
        total_samples = len(self.dataframe)
        positive_weight = (1.0 / positive_class_count) * (total_samples / 2.0)
        negative_weight = (1.0 / negative_class_count) * (total_samples / 2.0)
        self.class_weights = torch.Tensor([negative_weight, positive_weight])

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves and processes the data at the specified index.

        Args:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
          A tuple containing the input features (sequence, attention masks,
          categorical features) and target for the given index.
        """
        # Get encoded features
        sequence_values = self.dataframe[self._sequence].values[idx]
        tokens = self._tokenizer(
            sequence_values,
            return_tensors="pt",
            padding='max_length',
            max_length=MAX_SIZE)
        sequence_enc = (tokens["input_ids"].to(device)) 
        masks = (tokens["attention_mask"].to(device))

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
        target = torch.Tensor(target).float().to(device)

        return sequence_enc, masks, cat_features, target

    def get_weights(self):
        """
        Returns the class weights used for upsampling.

        Returns:
        - torch.Tensor: Class weights for upsampling.
        """
        return self.class_weights


def sample_weights(dataset):
    """
    Generate sample weights based on class weights for the given
    dataset.

    Args:
    - dataset (SeqCatBalancedDataset): The dataset for which sample 
      weights are generated.

    Returns:
    - list: A list of sample weights corresponding to each sample in
      the dataset.
    """
    cls_weights = dataset.get_weights()
    sample_weights = [
        cls_weights[0] if hit == 0 else cls_weights[1]
        for hit in dataset.dataframe['hit']
    ]
    return sample_weights
