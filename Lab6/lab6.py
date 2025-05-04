import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.svm import SVC


class RosenblattPerceptron:
    def __init__(self, random_weights=True, epochs=100, activation='step'):
        self.random_weights = random_weights
        self.epochs = epochs
        self.activation = activation
        self.weights = None
        self.bias = None
        self.errors_ = []

    def _initialize_weights(self, n_features):
        if self.random_weights:
            self.weights = np.random.rand(n_features)
            self.bias = np.random.rand()
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

    def _activate(self, x):
        if self.activation == 'step':
            return 1 if x >= 0 else 0
        elif self.activation == 'sign':
            return 1 if x >= 0 else -1
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError("Неизвестная функция активации")

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self._initialize_weights(n_features)

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(x, y):
                # Вычисление выхода
                net_input = np.dot(xi, self.weights) + self.bias
                output = self._activate(net_input)

                # Обновление весов
                update = (target - output)
                self.weights += update * xi
                self.bias += update

                errors += int(update != 0.0)

            self.errors_.append(errors)
            if errors == 0:
                break

    def predict(self, x):
        net_input = np.dot(x, self.weights) + self.bias
        return np.array([self._activate(xi) for xi in net_input])


n_samples = 500
data, labels = make_blobs(n_samples=n_samples,
                             centers=([1.1, 3], [4.5, 6.9]),
                             cluster_std=1.3,
                             random_state=0)


colours = ('green', 'orange')
fig, ax = plt.subplots()

for n_class in range(2):
    ax.scatter(data[labels==n_class][:, 0],
               data[labels==n_class][:, 1],
               c=colours[n_class],
               s=50,
               label=str(n_class))

perc = RosenblattPerceptron()
perc.fit(data, labels)
pred = perc.predict(data)
acc = accuracy_score(labels, pred)
print(f"Accuracy: {acc}")

# Визуализация разделяющей границы
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = perc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
for n_class in range(2):
    plt.scatter(data[labels == n_class][:, 0],
                data[labels == n_class][:, 1],
                c=colours[n_class],
                s=50,
                label=str(n_class))
plt.title("Разделяющая граница нашего перцептрона")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.show()


sk_perc = Perceptron(max_iter=100)
sk_perc.fit(data, labels)
sk_pred = sk_perc.predict(data)
sk_acc = accuracy_score(labels, sk_pred)
print(f"Accuracy (sklearn): {sk_acc}")


iris = load_iris()

X = iris.data[-100:]
Y = iris.target[-100:]

iris_perc = Perceptron(max_iter=100, random_state=0)
iris_perc.fit(X, Y)
iris_perc_pred = iris_perc.predict(X)
iris_perc_acc = accuracy_score(Y, iris_perc_pred)

svm = SVC(kernel='linear')
svm.fit(X, Y)
svm_pred = svm.predict(X)
svm_acc = accuracy_score(Y, svm_pred)

print(f"\nПерцептрон: {iris_perc_acc:.2f}")
print(f"SVM: {svm_acc:.2f}")
