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
