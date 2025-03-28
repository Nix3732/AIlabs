import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as mt
from matplotlib.patches import Rectangle

name = str(input('Введите путь файла: '))
data = pd.read_csv(name)

N = len(data)
print()
print(f'Количество строк: {N} \n')
print(f'Минимальное значение каждого столбца: \n{data.min()}\n')
print(f'Максимальное значение каждого столбца: \n{data.max()}\n')
print(f'Среднее значение каждого столбца: \n{data.mean()}\n')

col = data.columns.tolist()

flag = int(input(f"Какой столбец выберем для оси X: 1 - {col[0]}, 2 - {col[1]}?\n"))

if flag == 1:
    x = data[col[0]]
    y = data[col[1]]
    l = mt.ceil(data[col[0]].max() * 1.1)
else:
    x = data[col[1]]
    y = data[col[0]]
    l = mt.ceil(data[col[1]].max() * 1.1)

mx = x.sum()/N
my = y.sum()/N
a2 = np.dot(x.T, x)/N
a11 = np.dot(x.T, y)/N

k = (a11-mx*my)/(a2-mx**2)
b = my-k*mx
f = np.array([k*x+b for x in range(l)])

print(f'\nУравнение регрессии: y = {k:.2f}x + {b:.2f}')

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(x, y)
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(x, y)
plt.plot(f, c='red')
plt.grid(True)

ax3 = plt.subplot(1, 3, 3)
plt.scatter(x, y)
plt.plot(f, c='red')

for xi, yi in zip(x, y):
    y_p = k*xi+b
    error = yi - y_p
    side_length = abs(error)
    x_cor = 0
    y_cor = 0
    if error > 0: #верхние
        x_cor = xi-side_length*(max(x)/max(y))
        y_cor = y_p
    elif error < 0: #нижние
        x_cor = xi
        y_cor = yi

    square = Rectangle((x_cor, y_cor), side_length*(max(x)/max(y)), side_length, color='green', alpha=0.4
                       )
    ax3.add_patch(square)
plt.grid(True)
plt.show()