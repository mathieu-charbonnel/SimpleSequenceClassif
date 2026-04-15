import pandas as pd


def cleaning(df: pd.DataFrame) -> None:
    """Clean the allele column in-place: remove asterisks, extract class/gene/variant."""
    df["allele"] = df["allele"].replace(r"\*", "", regex=True)
    df["class"] = df["allele"].str[4]
    df["gene"] = df["allele"].str[5:7]
    df["variant"] = df["allele"].str[8:]
    df.drop("allele", axis=1, inplace=True)
