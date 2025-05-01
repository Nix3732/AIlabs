import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


def predict(X, k, b):
    return b + k * X


diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
col = df.columns.tolist()

X = df['bmi'].values.reshape(-1, 1)
Y = df['target'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Создание и обучение модели
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, Y_train)


# Обучение собственной модели
N = len(X_train)
mx = X_train.sum()/N
my = Y_train.sum()/N
a2 = np.dot(X_train.T, X_train)/N
a11 = np.dot(X_train.T, Y_train)/N

k = (a11-mx*my)/(a2-mx**2)
b = my-k*mx

print(f'\nУравнение регрессии: y = {float(k):.2f}x + {float(b):.2f}')

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Данные')

# Линия регрессии sklearn
plt.plot(X, sklearn_model.predict(X), color='red',
         label=f'Scikit-Learn: y = {sklearn_model.coef_[0]:.2f}x + {sklearn_model.intercept_:.2f}')

# Линия регрессии собственной модели
plt.plot(X, predict(X.ravel(), k, b).T, color='orange', linestyle='--',
         label=f'Моя модель: y = {float(k):.2f}x + {float(b):.2f}')

plt.xlabel('Индекс массы тела (BMI)')
plt.ylabel('Целевая переменная (диабет)')
plt.title('Линейная регрессия для набора данных diabetes')
plt.legend()
plt.grid(True)


# Создание таблицы с предсказаниями
results = pd.DataFrame({
    'Настоящие': Y_test,
    'Предсказания sklearn': sklearn_model.predict(X_test),
    'Мои предсказания': predict(X_test.ravel(), k, b)[0]
}, index=range(len(Y_test)))

print("\nТаблица с результатами предсказаний (первые 10 строк):")
print(results.head(10))


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return mae, r2, mape


y_pred_sklearn = sklearn_model.predict(X_test)

y_pred_custom = (X_test * k + b).flatten()

metrics_sklearn = metrics(Y_test, y_pred_sklearn)
metrics_custom = metrics(Y_test, y_pred_custom)

# Создаем таблицу для сравнения
metrics_table = pd.DataFrame({
    'Метрика': ['MAE', 'R²', 'MAPE'],
    'Scikit-Learn': [f"{metrics_sklearn[0]:.4f}", f"{metrics_sklearn[1]:.4f}",
                     f"{metrics_sklearn[2]:.2f}"],
    'Моя модель': [f"{metrics_custom[0]:.4f}", f"{metrics_custom[1]:.4f}",
                   f"{metrics_custom[2]:.2f}"]
})

print("\nСравнительная таблица метрик:")
print(metrics_table)

plt.show()
