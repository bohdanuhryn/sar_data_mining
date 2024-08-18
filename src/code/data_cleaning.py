from typing import List
from pandas import DataFrame

def apply_z_score(df: DataFrame, columns: List[str], threshold: int = 3) -> DataFrame:
  scoring_df = df[columns]
  
  z_scores = (scoring_df - scoring_df.mean()) / scoring_df.std()

  outliers = z_scores[(z_scores > threshold) | (z_scores < -threshold)].any(axis=1)

  outliers_data = df[outliers]

  print(f'\nOutliers count: {len(outliers_data)}')

  cleaned_data = df[~outliers]

  return cleaned_data