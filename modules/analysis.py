import pandas as pd


def df_category_intersect(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    feature: str,
) -> int:
    """Count unique values of a categorical feature common to both DataFrames."""
    combined = pd.concat([df0, df1], axis=0)
    unique0 = df0[feature].nunique()
    unique1 = df1[feature].nunique()
    unique_combined = combined[feature].nunique()
    return unique0 + unique1 - unique_combined


def max_length(df: pd.DataFrame, feature: str) -> int:
    """Return the maximum string length of a feature column."""
    return df[feature].apply(len).max()
