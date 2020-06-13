import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# print(load_digits())
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

y_5_train = (y_train == '5')
y_5_test = (y_test == '5')
# print(y_5_test)
sgd_clf = SGDClassifier(random_state=42)
forest_clf = RandomForestClassifier(random_state=42)
# sgd_clf.fit(X_train, y_5_train)
# y_predict = sgd_clf.predict(X_test)

y_score = cross_val_predict(sgd_clf, X_train, y_5_train, cv=3, method="decision_function")
y_probas_forest = cross_val_predict(forest_clf, X_train, y_5_train, cv=3, method="predict_proba")

y_score_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_5_train, y_score_forest)

# print("Prediction: ", y_score)
# print("Test: ", y_test)

# Precision and recal
# print(precision_score(y_5_train, y_score))
# print(recall_score(y_5_train, y_score))
