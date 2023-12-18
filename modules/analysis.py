import pandas as pd

def df_category_intersect(df0, df1, feature):
  """"
  Calculate the intersection of unique values for categorical 
  features between two DataFrames.

  Parameters:
  - df0 (pd.DataFrame): The first DataFrame.
  - df1 (pd.DataFrame): The second DataFrame.
  - feature (str): The name of the categorical feature for which the
  intersection is calculated.

  Returns:
  int: The count of unique values in the categorical feature that are 
  common to both DataFrames.
  """
  combined = pd.concat([df0, df1], axis=0)
  unique0 = df0[feature].nunique()
  unique1 = df1[feature].nunique()
  uniquec = combined[feature].nunique()
  return (unique0 + unique1 - uniquec)

def max_length(df, feature):
  """
  Calculate the maximum length of values in a specified feature of a
  DataFrame.

  Parameters:
  - df (pd.DataFrame): The DataFrame containing the data.
  - feature (str): The name of the feature

  Returns:
  int: The maximum length of values in the specified feature.
  """
  return df[feature].apply(lambda x: len(x)).max()