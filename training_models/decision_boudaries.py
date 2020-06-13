import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(iris.keys())
# print(iris['data'])
# Get just petal width from dataset

X = iris['data'][:, 3:]
y = (iris["target"] == 2).astype(np.int)
# print()
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict(X_new)
plt.plot(X_new, y_proba, "g-", label="Iris Virginica")
plt.plot(X_new, y_proba, "b--", label="Not Iris Virginica")
plt.show()
