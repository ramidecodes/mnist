import numpy as np

### Functions for you to fill in ###


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    kernel = np.dot(X, Y.T) + c
    kernel_matrix = kernel ** p
    return kernel_matrix
    raise NotImplementedError


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    distances = np.linalg.norm((X[:, None, :] - Y[:, :]), axis=2) ** 2
    kernel_matrix = np.exp(-gamma * distances)
    return kernel_matrix
    raise NotImplementedError


def quadratic_kernel(x):
    return np.array([x[0]**2, np.sqrt(2) * x[0] * x[1], x[1]**2])


def svm_quadratic_kernel(dataset, labels, mistakes):
    w = np.zeros(4)
    kernel_data = []

    for x in map(quadratic_kernel, dataset):
        kernel_data.append(x)

    np_kernel_data = np.array(kernel_data)

    t_data = np.hstack([np.ones(dataset.shape[0]).reshape(
        dataset.shape[0], 1), np_kernel_data])

    for i in range(dataset.shape[0]):
        w = w + mistakes[i] * labels[i] * t_data[i]

    theta_0 = w[0]
    theta = w[1:]
    return (theta_0, theta)
