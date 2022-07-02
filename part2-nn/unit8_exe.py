import numpy as np

# Unit 3 - Lecture 8 - Exercises
# Numerical Example


def basic_relu(X, W, w_0):

    z = w_0 + np.dot(X, W)
    print("Z: ", z)

    # Max between 0 & z
    # y = np.amax(z).clip(min=0)
    y = np.maximum(0, z)

    return y


def logic_nand(X):
    W = np.array([-2, -2])
    w_0 = 3
    # y = not(X[0] and X[1])
    # z = W[0]*X[0] + W[1]*X[1] + w_0
    z = np.dot(W, X) + w_0

    if (z <= 0):
        return 0
    else:
        return 1
