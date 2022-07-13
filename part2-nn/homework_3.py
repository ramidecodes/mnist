from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
from torch import conv2d


def relu(z):
    z_sum = np.linalg.norm(z) + z
    return z_sum / 2


def simple_nn(x_1, x_2, w, w_0, v, v_01, v_02):
    # TODO: Implement solution using straight matrix operations

    b = 1

    z_1 = x_1 * w[0][0] + x_2 * w[0][1] + w_0
    z_2 = x_1 * w[1][0] + x_2 * w[1][1] + w_0
    z_3 = x_1 * w[2][0] + x_2 * w[2][1] + w_0
    z_4 = x_1 * w[3][0] + x_2 * w[3][1] + w_0

    print("Z1: ", z_1)
    print("Z2: ", z_2)
    print("Z3: ", z_3)
    print("Z4: ", z_4)

    u_1 = relu(z_1) * v[0][0] + relu(z_2) * v[0][1] + \
        relu(z_3) * v[0][2] + relu(z_4) * v[0][3] + v_01
    u_2 = relu(z_1) * v[1][0] + relu(z_2) * v[1][1] + \
        relu(z_3) * v[1][2] + relu(z_4) * v[1][3] + v_02

    # u_1 = 1 * v[0][0] + 1 * v[0][1] + 2 * v[0][2] - 1 * v[0][3] + v_01
    # u_2 = 1 * v[1][0] + 1 * v[1][1] + 2 * v[1][2] - 1 * v[1][3] + v_02

    # u_exp_sum = np.exp(relu(u_1)) + np.exp(relu(u_2))

    # o_1 = np.exp(relu(u_1)) / u_exp_sum
    # o_2 = np.exp(relu(u_2)) / u_exp_sum

    u_exp_sum = np.exp(b * relu(u_1)) + np.exp(b * relu(u_2))

    o_1 = np.exp(b * relu(u_1)) / u_exp_sum
    o_2 = np.exp(b * relu(u_2)) / u_exp_sum

    return (o_1, o_2)


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


def quadratic_loss(y, t):
    c = ((y - t) ** 2) / 2
    return c


def simple_lstm(x, h, c, w_fh, w_fx, w_ih, w_ix, w_oh, w_ox, w_ch, w_cx, b_f, b_i, b_o, b_c):

    f = sigmoid(w_fh * h + w_fx * x + b_f)
    i = sigmoid(w_ih * h + w_ix * x + b_i)
    o = sigmoid(w_oh * h + w_ox * x + b_o)

    c = f * c + i * np.tanh(w_ch * h + w_cx * x + b_c)

    h = o * np.tanh(c)

    return (c, round(h))


def two_layer_nn(t, x, w_1, w_2, b):
    z_1 = w_1 * x
    a_1 = relu(z_1)
    z_2 = w_2 * a_1 + b
    y = sigmoid(z_2)
    c = quadratic_loss(y, t)
    return c


def simple_convolutional_nn(f, g):
    conv_valid = np.convolve(f, g, "valid")  # No padding
    conv_same = np.convolve(f, g, "same")  # Using padding
    return (conv_valid, conv_same)


def conv2d(f, g):
    conv = signal.convolve2d(f, g, "valid")
    return conv
