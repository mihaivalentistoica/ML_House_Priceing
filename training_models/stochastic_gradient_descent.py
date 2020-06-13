import numpy as np
import time

eta = 1
n_interations = 1000
n_epochs = 50
m = 100
t0, t1 = 5, 50
theta = np.random.randn(2, 1)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]


def learning_schedule(t):
    return t0 / (t + t1)


start_time = time.time()
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index + 1]
        yi = y[random_index: random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        print("Eta: ", eta)
        theta = theta - eta * gradients

print("Time: ", time.time() - start_time)
print(theta)
