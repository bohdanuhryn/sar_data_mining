from pandas import DataFrame

def get_z_score(data: DataFrame, threshold: int = 3):
    print("\nZ-score:")
    z_scores = (data - data.mean()) / data.std()

    outliers = z_scores[(z_scores > threshold) | (z_scores < -threshold)].any(axis=1)

    outliers_data = data[outliers]

    print(f'\nOutliers count: {len(outliers_data)}')

    cleaned_data = data[~outliers]

    return cleaned_data