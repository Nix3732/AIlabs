import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import math as mt
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
N = len(X)

# Создание и обучение модели
sklearn_model = LinearRegression()
sklearn_model.fit(X, Y)


# Обучение собственной модели
mx = X.sum()/N
my = Y.sum()/N
a2 = np.dot(X.T, X)/N
a11 = np.dot(X.T, Y)/N

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
    'Настоящие': Y,
    'Предсказания sklearn': sklearn_model.predict(X),
    'Мои предсказания': predict(X.ravel(), k, b)[0]
}, index=range(len(Y)))

print("\nТаблица с результатами предсказаний (первые 10 строк):")
print(results.head(10))

plt.show()