import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

df_main = pd.read_csv('Titanic.csv')
value_data = len(df_main)

df = df_main.dropna()

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

df = df.drop('PassengerId', axis=1)

value_data_clean = len(df)

percent = (1-(value_data_clean / value_data))*100

print(f'Процент потерянных данных: {percent:.4f}%')

X = df.drop('Survived', axis=1)
Y = df['Survived']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=0)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
acc1 = accuracy_score(Y_pred, Y_test)

print(f"Точность модели: {acc1:.4f}")

df = df.drop('Embarked', axis=1)

X = df.drop('Survived', axis=1)
Y = df['Survived']


clf = LogisticRegression(max_iter=1000, random_state=0)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
acc2 = accuracy_score(Y_test, Y_pred)

print(f"Точность модели: {acc2:.4f}")

print(f"Влияние Embarked: {acc2-acc1:.4f}")


pr = precision_score(Y_test, Y_pred)
rec = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
print(f"\nМетрика Precision: {pr:.4f}")
print(f"Метрика Recall: {rec:.4f}")
print(f"Метрика F1: {f1:.4f}")

matrix = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Истинный класс')
plt.ylabel('Предсказанный класс')
plt.show()

Y_proba = clf.predict_proba(X_test)[:, 1]

precision_vals, recall_vals, thresholds = precision_recall_curve(Y_test, Y_proba)
plt.plot(recall_vals, precision_vals, linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.grid(True)
plt.show()


fpr, tpr, thresholds = roc_curve(Y_test, Y_proba)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid()
plt.show()

#Часть 2

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, Y_train)
Y_pred_svm = svm_model.predict(X_test)
Y_proba_svm = svm_model.predict_proba(X_test)[:, 1]

pr_svm = precision_score(Y_test, Y_pred_svm)
rec_svm = recall_score(Y_test, Y_pred_svm)
f1_svm = f1_score(Y_test, Y_pred_svm)

print(f"\nМетрика Precision (SVM): {pr_svm:.4f}")
print(f"Метрика Recall (SVM): {rec_svm:.4f}")
print(f"Метрика F1 (SVM): {f1_svm:.4f}")

matrix_svm = confusion_matrix(Y_test, Y_pred_svm)
plt.figure(figsize=(10, 6))
sns.heatmap(matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Истинный класс')
plt.ylabel('Предсказанный класс')
plt.show()


precision_vals_svm, recall_vals_svm, thresholds_svm = precision_recall_curve(Y_test, Y_proba_svm)
plt.plot(recall_vals_svm, precision_vals_svm, linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.grid(True)
plt.show()


fpr_svm, tpr_svm, thresholds_svm = roc_curve(Y_test, Y_proba_svm)
plt.plot(fpr_svm, tpr_svm)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid()
plt.show()


knc = KNeighborsClassifier()
knc.fit(X_train, Y_train)
Y_pred_knc = knc.predict(X_test)
Y_proba_knc = knc.predict_proba(X_test)[:, 1]

pr_knc = precision_score(Y_test, Y_pred_knc)
rec_knc = recall_score(Y_test, Y_pred_knc)
f1_knc = f1_score(Y_test, Y_pred_knc)

print(f"\nМетрика Precision (KNC): {pr_knc:.4f}")
print(f"Метрика Recall (KNC): {rec_knc:.4f}")
print(f"Метрика F1 (KNC): {f1_knc:.4f}")

matrix_knc = confusion_matrix(Y_test, Y_pred_knc)
plt.figure(figsize=(10, 6))
sns.heatmap(matrix_knc, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Истинный класс')
plt.ylabel('Предсказанный класс')
plt.show()


precision_vals_knc, recall_vals_knc, thresholds_knc = precision_recall_curve(Y_test, Y_proba_knc)
plt.plot(recall_vals_knc, precision_vals_knc, linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.grid(True)
plt.show()


fpr_knc, tpr_knc, thresholds_knc = roc_curve(Y_test, Y_proba_knc)
plt.plot(fpr_knc, tpr_knc)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid()
plt.show()

print(f"Лучшая модель по метрике f1 - SVM")
