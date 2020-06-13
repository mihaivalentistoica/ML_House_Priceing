import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)
# print("X: ", X)
# print("y: ", y)

X_b = np.c_[np.ones((100, 1)), X]  # Atribuire valoare X0 = 1, vezi in caiet
theta_bet = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_bet)

# Facem o prezicere cu ajutorul lui theta_best

X_new = np.array(([[0], [2]]))
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Atribuie x0 = 1
y_predict = X_new_b.dot(theta_bet)

print("X_new: ", X_new)
print("Y_predict: ", y_predict)
print('\n')
print("X_new_b: ", X_new_b)

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()
