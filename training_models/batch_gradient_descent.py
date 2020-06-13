import numpy as np
import time

eta = 0.1
n_interations = 1000
m = 100
theta = np.random.randn(2, 1)

print("theta befor: ", theta)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

start_time = time.time()
for interation in range(n_interations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print("time: ", time.time() - start_time)
print("theta after: ", theta)
