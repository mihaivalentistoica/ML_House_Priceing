import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
sgd_clf = SGDClassifier()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

sgd_clf.fit(X_train_scaled, y_train)
# print("Prediction: ", sgd_clf.predict(X_test))

# prediction = sgd_clf.predict(X_test)
# decision = sgd_clf.decision_function(X_test)

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
cross_score = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

# with open('files/prediction.txt', 'w') as pred_file:
#     for p, index in prediction:
#         pred_file.write(p)
#         pred_file.write(' : ')
#         pred_file.write(y_test[index])
#         pred_file.write('\n')

# with open('files/decision.txt', 'w') as dec_file:
#     for d in decision:
#         dec_file.write(d)
#         dec_file.write('\n')
# print()
# print(decision)
