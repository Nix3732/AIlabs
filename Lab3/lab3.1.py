from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
col = df.columns.tolist()

plt.figure(figsize=(10, 6))
index = set(iris.target)
color = ['r', 'g', 'b']


plt.subplot(1, 2, 1)
for target, clr in zip(index, color):
    plt.scatter(df[df['target'] == target][col[0]], df[df['target'] == target][col[1]],
                c=clr, label=iris.target_names[target])
plt.xlabel(col[0])
plt.ylabel(col[1])
plt.title(f"{col[0]} vs {col[1]}")
plt.legend()


plt.subplot(1, 2, 2)
for target, clr in zip(index, color):
    plt.scatter(df[df['target'] == target][col[2]], df[df['target'] == target][col[3]],
                c=clr, label=iris.target_names[target])
plt.xlabel(col[2])
plt.ylabel(col[3])
plt.title(f"{col[2]} vs {col[3]}")
plt.legend()
plt.tight_layout()

sns.pairplot(df, hue='target')


df1 = df[df['target'].isin([0, 1])]
df2 = df[df['target'].isin([1, 2])]

X1 = df1.drop('target', axis=1)
Y1 = df1['target']

X2 = df2.drop('target', axis=1)
Y2 = df2['target']

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.2, random_state=42)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2, random_state=42)


clf1 = LogisticRegression(random_state=0)
clf2 = LogisticRegression(random_state=0)

clf1.fit(X_train1, Y_train1)
clf2.fit(X_train2, Y_train2)

Y_pred1 = clf1.predict(X_test1)
Y_pred2 = clf2.predict(X_test2)

acc1 = accuracy_score(Y_test1, Y_pred1)
acc2 = accuracy_score(Y_test2, Y_pred2)

print('Score1:', acc1)
print('Score2:', acc2)


X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=1,
                           n_clusters_per_class=1)


df_r = pd.merge(pd.DataFrame(X), pd.DataFrame(Y), left_index=True, right_index=True)
df_r = df_r.rename(columns={'0_x': 'X1', '1': 'X2', '0_y': 'Y'})
dfrcol = df_r.columns.tolist()


plt.figure(figsize=(10, 6))
for target, clr in zip(index, color):
    plt.scatter(df_r[df_r['Y'] == target][dfrcol[0]], df_r[df_r['Y'] == target][dfrcol[1]], c=clr)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Сгенерированный датасет')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_pred, Y_test)

print(acc)

plt.show()
