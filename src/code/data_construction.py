from pandas import DataFrame
from typing import List

def add_moving_averages(df: DataFrame, columns: List[str], window: int = 7):
    for column in columns:
        df[f'{column}_ma'] = df[column].rolling(window=window).mean()

    return df