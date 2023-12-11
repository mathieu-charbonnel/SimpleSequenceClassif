def cleaning(df):
  df['allele'] = df['allele'].replace(r'\*', '', regex=True)
  df['class'] = df['allele'].str[4]
  df['gene'] = df['allele'].str[5:7]
  df['variant'] = df['allele'].str[8:]
  df = df.drop('allele', axis=1, inplace=True)