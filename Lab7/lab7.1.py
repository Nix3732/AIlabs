import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class BostonHousingModel:
    def __init__(self):
        self.results = []
        warnings.filterwarnings("ignore", category=UserWarning)

    def load_data(self):
        self.data = pd.read_csv('BostonHousing.csv')
        self.X = self.data.drop('medv', axis=1)
        self.y = self.data['medv']

    def preprocess_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, architecture, model_name=None):
        model = MLPRegressor(
            hidden_layer_sizes=architecture,
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )

        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)

        metrics = {
            "Архитектура": str(architecture),
            "MSE": mean_squared_error(self.y_test, y_pred),
            "R2": r2_score(self.y_test, y_pred),
            "Итерации": model.n_iter_
        }

        if model_name:
            self._print_single_result(model_name, metrics)

        return metrics

    def _print_single_result(self, title, metrics):
        print(f"\n{title}")
        print("=" * 50)
        df = pd.DataFrame({
            "Metric": ["Mean Squared Error", "R-squared"],
            "Value": [metrics["MSE"], metrics["R2"]]
        })
        print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".4f"))
        print("=" * 50)

    def evaluate_architectures(self, architectures):
        print("\nИССЛЕДОВАНИЕ АРХИТЕКТУР СЕТИ")
        print("=" * 50)

        for arch in architectures:
            self.results.append(self.train_model(arch))

        print(tabulate(
            pd.DataFrame(self.results),
            headers="keys",
            tablefmt="pretty",
            showindex=False,
            floatfmt=".4f"
        ))

    def plot_results(self):
        plt.figure(figsize=(14, 6))

        # График R2
        plt.subplot(1, 2, 1)
        bars = plt.bar(
            [x['Архитектура'] for x in self.results],
            [x['R2'] for x in self.results],
            color='skyblue'
        )
        self._configure_plot(
            'Качество моделей (R-squared)',
            'Архитектура сети',
            'R-squared',
            bars,
            (0, 1)
        )

        # График MSE
        plt.subplot(1, 2, 2)
        bars = plt.bar(
            [x['Архитектура'] for x in self.results],
            [x['MSE'] for x in self.results],
            color='lightgreen'
        )
        self._configure_plot(
            'Ошибка моделей (MSE)',
            'Архитектура сети',
            'MSE',
            bars
        )

        plt.tight_layout()
        plt.show()

    def _configure_plot(self, title, xlabel, ylabel, bars, ylim=None):
        """Настройка внешнего вида графиков"""
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if ylim:
            plt.ylim(ylim)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}' if ylabel == 'R-squared' else f'{height:.1f}',
                ha='center',
                va='bottom'
            )

    def print_best_model(self):
        """Вывод информации о лучшей модели"""
        best_model = max(self.results, key=lambda x: x['R2'])
        print("\nЛУЧШАЯ АРХИТЕКТУРА СЕТИ:")
        print("=" * 50)
        print(f"Архитектура: {best_model['Архитектура']}")
        print(f"MSE: {best_model['MSE']:.4f}")
        print(f"R2: {best_model['R2']:.4f}")
        print(f"Итераций обучения: {best_model['Итерации']}")
        print("=" * 50)


def main():
    housing_model = BostonHousingModel()

    housing_model.load_data()
    housing_model.preprocess_data()

    housing_model.train_model((100,), "БАЗОВАЯ МОДЕЛЬ (100 нейронов в 1 слое)")

    architectures = [
        (10,),
        (50,),
        (100,),
        (50, 50),
        (100, 50, 25),
        (150, 100, 50)
    ]

    housing_model.evaluate_architectures(architectures)

    housing_model.plot_results()

    housing_model.print_best_model()


if __name__ == "__main__":
    main()
