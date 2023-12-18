def cleaning(df):
    """
    Perform static data cleaning specifically for my medical private
    dataset.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing allele information.

    Returns:
    None: The cleaning is done in-place, and the original DataFrame is
    modified.
    """
    # Remove asterisks from the 'allele' column
    df['allele'] = df['allele'].replace(r'\*', '', regex=True)

    # Extract information into separate columns
    df['class'] = df['allele'].str[4]
    df['gene'] = df['allele'].str[5:7]
    df['variant'] = df['allele'].str[8:]

    # Drop the original 'allele' column
    df.drop('allele', axis=1, inplace=True)
