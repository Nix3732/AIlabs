import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score,
                             precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DigitClassifier:
    def __init__(self):
        self.digits = load_digits()
        self.model = None
        self.scaler = StandardScaler()
        self.class_report = None
        self.cm = None

    def visualize_samples(self, n_samples=10):
        plt.figure(figsize=(12, 4))
        plt.suptitle("Примеры изображений цифр из датасета", fontsize=14, y=1.05)

        for index, (image, label) in enumerate(zip(self.digits.data[:n_samples],
                                                   self.digits.target[:n_samples])):
            plt.subplot(2, 5, index + 1)
            plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray_r)
            plt.title(f'Цифра: {label}', fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def prepare_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.digits.data,
            self.digits.target,
            test_size=test_size,
            random_state=random_state,
            stratify=self.digits.target
        )

        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, hidden_layer_sizes=(150, 100), max_iter=1000):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            verbose=True
        )

        print("\nОбучение модели...")
        self.model.fit(self.X_train_scaled, self.y_train)
        print(f"Обучение завершено. Количество итераций: {self.model.n_iter_}")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test_scaled)
        y_proba = self.model.predict_proba(self.X_test_scaled)

        metrics = [
            ["Accuracy", accuracy_score(self.y_test, y_pred)],
            ["Precision (weighted)", precision_score(self.y_test, y_pred, average='weighted')],
            ["Recall (weighted)", recall_score(self.y_test, y_pred, average='weighted')]
        ]

        self._print_metrics("ОСНОВНЫЕ МЕТРИКИ МОДЕЛИ:", metrics)

        self.class_report = classification_report(self.y_test, y_pred, output_dict=True)
        print("\nПОДРОБНЫЙ ОТЧЕТ ПО КЛАССАМ:")
        print("=" * 50)
        print(classification_report(self.y_test, y_pred, digits=4))
        print("=" * 50)

        self.cm = confusion_matrix(self.y_test, y_pred)
        self._plot_confusion_matrix()

        self._print_class_metrics()
        self._plot_class_metrics()

    def _print_metrics(self, title, metrics):
        print(f"\n{title}")
        print("=" * 50)
        print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid", floatfmt=".4f"))
        print("=" * 50)

    def _plot_confusion_matrix(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10),
                    yticklabels=range(10))
        plt.title('Матрица ошибок', fontsize=14, pad=20)
        plt.xlabel('Предсказанные метки', fontsize=12)
        plt.ylabel('Истинные метки', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

    def _print_class_metrics(self):
        class_metrics = []
        for digit in range(10):
            class_metrics.append([
                digit,
                self.class_report[str(digit)]['precision'],
                self.class_report[str(digit)]['recall'],
                self.class_report[str(digit)]['f1-score'],
                self.class_report[str(digit)]['support']
            ])

        print("\nМЕТРИКИ ПО КЛАССАМ:")
        print("=" * 90)
        print(tabulate(class_metrics,
                       headers=["Цифра", "Precision", "Recall", "F1-Score", "Поддержка"],
                       tablefmt="grid",
                       floatfmt=".4f"))
        print("=" * 90)

    def _plot_class_metrics(self):
        plt.figure(figsize=(14, 6))
        plt.bar(range(10), [self.class_report[str(d)]['precision'] for d in range(10)],
                color='skyblue', alpha=0.7, label='Precision')
        plt.bar(range(10), [self.class_report[str(d)]['recall'] for d in range(10)],
                color='lightgreen', alpha=0.7, label='Recall', width=0.4)

        plt.title('Точность и полнота по классам', fontsize=14)
        plt.xlabel('Класс (цифра)', fontsize=12)
        plt.ylabel('Значение метрики', fontsize=12)
        plt.xticks(range(10))
        plt.ylim(0.85, 1.0)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i in range(10):
            plt.text(i - 0.15, self.class_report[str(i)]['precision'] + 0.01,
                     f"{self.class_report[str(i)]['precision']:.3f}", fontsize=9)
            plt.text(i + 0.15, self.class_report[str(i)]['recall'] + 0.01,
                     f"{self.class_report[str(i)]['recall']:.3f}", fontsize=9)

        plt.tight_layout()
        plt.show()


def main():
    classifier = DigitClassifier()

    classifier.visualize_samples()

    classifier.prepare_data()

    classifier.train_model()

    classifier.evaluate_model()


if __name__ == "__main__":
    main()