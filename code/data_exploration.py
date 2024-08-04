from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sn

def show_boxplots(data: DataFrame):
    # Визначаємо числові колонки
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Визначаємо кількість рядків і колонок для підграфіків
    n_cols = 5
    n_rows = (len(numerical_columns) + 1) // n_cols

    # Створюємо boxplots для кожної числової ознаки
    plt.figure(figsize=(15, n_rows * 5))

    for i, column in enumerate(numerical_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sn.boxplot(y=column, data=data)
        plt.title(column)
        plt.tight_layout()

    plt.show()

def show_heatmap(data: DataFrame):
    dataForCorr = data[[
                        'launch_time',
                        'draw_time_avg',
                        'draw_time_median',
                        'total_frames',
                        'total_frames_median',
                        'janky_frames',
                        'janky_frames_median',
                        'janky_ratio',
                        'janky_ratio_median',
                        'missed_vsync_count',
                        'high_input_latency_count',
                        'slow_ui_thread_count',
                        'slow_bitmap_uploads_count',
                        'slow_issue_draw_commands_count'
                        ]]

    #corrMatrix = dataForCorr.corr(method="pearson")
    #corrMatrix = dataForCorr.corr(method="kendall")
    corrMatrix = dataForCorr.corr(method="spearman")

    plt.figure(figsize=(10, 8))

    sn.heatmap(corrMatrix, annot=True)

    plt.show()