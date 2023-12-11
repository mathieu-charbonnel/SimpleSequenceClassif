import pandas as pd

def df_category_intersect(df0, df1, feature):
  combined = pd.concat([df0, df1], axis=0)
  unique0 = df0[feature].nunique()
  unique1 = df1[feature].nunique()
  uniquec = combined[feature].nunique()
  return (unique0 + unique1 - uniquec)

def max_length(df, feature):
  return(df[feature].apply(lambda x: len(x)).max())