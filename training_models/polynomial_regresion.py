import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", label="train")
    plt.plot(np.sqrt(val_errors), "b-", label="val")
    plt.show()


X_gen = 2 * np.random.rand(100, 1)
y_gen = 4 + 3 * X_gen + np.random.rand(100, 1)

lin_reg = LinearRegression()
polynomial_regresion = Pipeline([
    ("poly_feauture", PolynomialFeatures(degree=10, include_bias=False)),
    ('sgd_reg', LinearRegression())
])
# poly_feautes = PolynomialFeatures(degree=10, include_bias=False)
#
# X_gen = poly_feautes.fit_transform(X_gen)

plot_learning_curves(polynomial_regresion, X_gen, y_gen)