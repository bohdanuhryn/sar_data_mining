import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from typing import Dict
from pandas import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, make_scorer

# Підготовка даних для моделі (створюємо лаги)
def create_lags(
    df: DataFrame,
    column: str,
    n_lags: int = 1
) -> None:
    df_copy = df.copy()
    for i in range(1, n_lags + 1):
        df_copy[f'lag_{i}'] = df_copy[column].shift(i)
    df_copy.dropna(inplace=True)
    return df_copy

def cross_validation_evaluator(model: BaseEstimator, x, y, cv) -> Dict[str, BaseEstimator]:
    mse_scores = cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error')
    mae_scores = cross_val_score(model, x, y, cv=cv, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, x, y, cv=cv, scoring='r2')
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    mape_scores = cross_val_score(model, x, y, cv=cv, scoring=mape_scorer)

    mse = -mse_scores.mean()
    rmse = np.sqrt(-mse_scores.mean())
    mae = -mae_scores.mean()
    r2 = r2_scores.mean()
    mape = -mape_scores.mean()

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

# Порівняння регресійних алгоритмів
def find_best_regression_algorithm(
    df: DataFrame,
    data_feature: str,
    target_feature: str
) -> BaseEstimator:
    # Вибрані колонки характеристик та прогнозування
    selected_columns = [data_feature, target_feature]

    # Відфільтрований набір даних
    df = df[selected_columns].copy()

    df.set_index(data_feature, inplace=True)

    # Кількість лагів
    n_lags = 20

    # Оновити набір даних з лагами
    df = create_lags(df, target_feature, n_lags)

    # Визначаємо точку розбиття (наприклад, 80% для тренувального набору, 20% для тестового)
    split_point = int(len(df) * 0.8)

    # Розділяємо дані
    train, test = df.iloc[:split_point], df.iloc[split_point:]

    # Тренувальні дані з лагами
    X_train, y_train = train.drop(target_feature, axis=1), train[target_feature]

    X_train_len = len(X_train)

    # Тестові дані з лагами
    X_test, y_test = test.drop(target_feature, axis=1), test[target_feature]

    # Набір моделей з параметрами за замовчуванням
    models: Dict[str, BaseEstimator] = defaultEstimatorModels()

    # Time Series Split for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Evaluate models using cross-validation
    cv_results: Dict[str, Dict[str, any]] = {}

    for name, model in models.items():
        cv_results[name] = cross_validation_evaluator(model, X_train, y_train, tscv)

    print('CV results:')
    cv_results_df = pd.DataFrame(cv_results).T
    print(cv_results_df)

    # Вибір найкращої моделі за метрикою R²
    best_models: Dict[str, BaseEstimator] = {name: model for name, model in models.items() if cv_results[name]['R2'] > 0}

    for name, model in best_models.items():
        # Навчання найкращої моделі на навчальній вибірці
        model.fit(X_train, y_train)

        # Останні відомі значення для створення початкових лагів
        last_known_values = df[target_feature].values[(X_train_len - n_lags):X_train_len]
        print(f'last_known_values: {last_known_values}')

        n_steps = len(X_test)

        # Передбачення на тестовій вибірці
        #predictions = best_model.predict(X_test)
        predictions = forecast_next_values(model, last_known_values, n_steps, n_lags)

        # Оцінка моделі на тестовій вибірці
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

        # Виведення метрик
        best_model_results = {
                'MSE': [mse],
                'RMSE': [rmse],
                'MAE': [mae],
                'R2': [r2],
                'MAPE': [mape]
        }
        print(f'Best Model: {name}')
        best_model_results_df = pd.DataFrame(best_model_results).T
        print(best_model_results_df)

        # Побудова графіку фактичних та прогнозованих значень
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df[target_feature], label='Actual')
        plt.plot(X_train.index, model.predict(X_train), label='Trained')
        plt.plot(y_test.index, predictions, label='Predicted', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Actual vs Predicted Values ({name})')
        plt.legend()
        plt.show()

    return best_models

# Прогнозування на нових даних
def forecast_next_values(model, last_known_values, n_steps, n_lags):
    predictions = []
    current_values = last_known_values.copy()

    for _ in range(n_steps):
        # Створення лагів для поточного набору даних
        lagged_values = current_values[-n_lags:].reshape(1, -1)
        # Прогнозування наступного значення
        next_value = model.predict(lagged_values)[0]
        predictions.append(next_value)
        # Додавання нового прогнозованого значення до набору даних
        current_values = np.append(current_values, next_value)

    return predictions

def forecast_aging(
    model: BaseEstimator,
    df: DataFrame,
    data_feature: str,
    target_feature: str,
    n_steps: int = 100
) -> None:
    selected_columns = [data_feature, target_feature]

    df = df[selected_columns].copy()

    n_lags = 3

    df = create_lags(df, target_feature, n_lags)

    X, y = df.drop(target_feature, axis=1), df[target_feature]

    # Передбачення на тестовій вибірці
    #predictions = model.predict(X)

    # Останні відомі значення для створення початкових лагів
    last_known_values = df[target_feature].values[-n_lags:]

    # Прогнозування наступних значень
    predictions = forecast_next_values(model, last_known_values, n_steps, n_lags)

    # Оцінка моделі на тестовій вибірці
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)

    # Виведення метрик
    model_results = {
        'MSE': [mse],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2],
        'MAPE': [mape]
    }
    model_results_df = pd.DataFrame(model_results).T
    print(model_results_df)

    # Побудова графіку фактичних та прогнозованих значень
    plt.figure(figsize=(14, 7))
    plt.plot(X, y, label='Actual')
    plt.plot(X, predictions, label='Predicted', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted Values')
    plt.legend()
    plt.show()
