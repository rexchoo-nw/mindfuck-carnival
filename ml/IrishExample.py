# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
print('done importing')


# %%
data = pd.read_csv('./dataset/iris.csv')
data.head()


# %%
data.info()


# %%
data.describe()


# %%
data['species'].value_counts()


# %%
plotting = sns.pairplot(data, hue='species')


# %%
x = data.drop(['species'], axis=1)


# %%
y = data['species']


# %%
from sklearn.model_selection import train_test_split


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)


# %%
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# %%
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()


# %%
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))


# %%
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(x, y)
knn.predict([[6, 3, 4, 2]])


# %%



