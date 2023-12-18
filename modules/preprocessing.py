import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def categories_fit_one_hot(df, categories):
    """
    Fit one-hot encoders for specified categorical columns in a
    DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categories (list): A list of column names representing 
      categorical features

    Returns:
    - Tuple[np.ndarray, List[OneHotEncoder]]: A tuple containing 
      the concatenated one-hot encoded features and a list of fitted 
      OneHotEncoder objects.
    """
    full_encoding = np.empty((df.shape[0], 0))
    encoders = []

    for category in categories:
        cat_values = df[category].values
        cat_encoder = OneHotEncoder(sparse_output=False,
                                    categories='auto',
                                    handle_unknown='ignore')
        encoded_cat = cat_encoder.fit_transform(cat_values.reshape(-1, 1))

        # Concatenate the one-hot encoded features
        full_encoding = np.hstack((full_encoding, encoded_cat))
        encoders.append(cat_encoder)

    return full_encoding, encoders

def categories_transform_one_hot(df, categories, encoders):
    """
    Transform specified categorical columns in a DataFrame using
    pre-fitted one-hot encoders.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categories (list): A list of column names representing 
      categorical features.
    - encoders (list): A list of pre-fitted OneHotEncoder objects

    Returns:
    - np.ndarray: The concatenated one-hot encoded features
    """
    full_encoding = np.empty((df.shape[0], 0))
    for category, encoder in zip(categories, encoders):
        cat_values = df[category].values
        encoded_cat = encoder.transform(cat_values.reshape(-1, 1))

        # Concatenate the one-hot encoded features
        full_encoding = np.hstack((full_encoding, encoded_cat))

    return full_encoding

def seq_one_hot(strng, alphabet, max_size):
    """
    One-hot encode a sequence string based on a given alphabet

    Parameters:
    - strng (str): The input sequence string
    - alphabet (str): The set of characters in the alphabet
    - max_size (int): The maximum size of the one-hot encoded vector

    Returns:
    - List[List[int]]: A list of lists representing the one-hot encoded
      vector
    """
    strng = f"{strng:<{max_size}}"
    token = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return token

def seq_pipeline(strng, alphabet, max_size):
    """
    Process a sequence string through one-hot encoding and convert to
    PyTorch tensor.

    Parameters:
    - strng (str): The input sequence string
    - alphabet (str): The set of characters in the alphabet
    - max_size (int): The maximum size of the one-hot encoded vector

    Returns:
    - torch.Tensor: The one-hot encoded sequence as a PyTorch tensor
    """
    token = seq_one_hot(strng, alphabet, max_size)
    token = torch.Tensor(token).reshape((1, -1))
    return token.to(device)

def df_seq_onehot_encode(df, sequence, alphabet, max_size):
    """
    One-hot encode a sequence column in a DataFrame

    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - sequence (str): The column name representing the sequence data
    - alphabet (str): The set of characters in the alphabet
    - max_size (int): The maximum size of the one-hot encoded vector

    Returns:
    - np.ndarray: The one-hot encoded sequences as a flattened array
    """
    X = df[sequence].values
    res = np.array([seq_one_hot(aa_sequence, alphabet, max_size) 
                   for aa_sequence in X])
    return res.reshape((res.shape[0], -1))
