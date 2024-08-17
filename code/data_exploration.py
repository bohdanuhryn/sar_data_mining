from typing import List
from typing import Optional
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sn

def show_time_series_charts(df: DataFrame):
    # Визначаємо числові колонки
    numerical_columns = df.select_dtypes(include=[int, float]).columns

    # Визначаємо кількість рядків і колонок для підграфіків
    n_cols = min(3, len(numerical_columns))
    n_rows = (len(numerical_columns) + 1) // n_cols

    # Створюємо графіки для кожної числової ознаки
    plt.figure(figsize=(16, n_rows * 4))

    for i, column in enumerate(numerical_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(df[column])
        plt.title(column)
        plt.tight_layout()

    plt.show()

def show_boxplots(df: DataFrame):
    # Визначаємо числові колонки
    numerical_columns = df.select_dtypes(include=[int, float]).columns

    # Визначаємо кількість рядків і колонок для підграфіків
    n_cols = min(4, len(numerical_columns))
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols

    # Створюємо boxplots для кожної числової ознаки
    plt.figure(figsize=(16, n_rows * 4))

    for i, column in enumerate(numerical_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sn.boxplot(y=column, data=df)
        plt.title(column)
        
    plt.tight_layout()
    plt.show()

def show_distributions(df: DataFrame):
    # Визначаємо числові колонки
    numerical_columns = df.select_dtypes(include=[int, float]).columns

    # Визначаємо кількість рядків і колонок для підграфіків
    n_cols = min(4, len(numerical_columns))
    n_rows = (len(numerical_columns) + 1) // n_cols
    
    # fig = plt.figure(figsize=(16, n_rows * 4))
    # gs = gridspec.GridSpec(n_rows, n_cols)

    # Створюємо boxplots для кожної числової ознаки
    plt.figure(figsize=(16, n_rows * 4))

    for i, column in enumerate(numerical_columns):
        # x = i // n_cols
        # y = i % n_cols
        # ax = fig.add_subplot(gs[x, y])
        plt.subplot(n_rows, n_cols, i + 1)
        # ax.hist(df[column], density=True)
        # ax.set_title(f'Distribution of {column}')
        # ax.set_xlabel(column)
        # ax.set_ylabel('Density')
        sn.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.tight_layout()
        
    plt.show()

def show_heatmap(df: DataFrame, columns: Optional[List[str]] = None) -> None:
    if columns is not None:
        df = df[columns]
    
    df = df.select_dtypes(include=[int, float])

    corrMatrix = df.corr(method="spearman")

    plt.figure(figsize=(10, 8))

    sn.heatmap(corrMatrix, annot=True)

    plt.show()