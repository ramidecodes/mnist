import numpy as np


def relu(z):
    z_sum = np.linalg.norm(z) + z
    return z_sum / 2


def simple_nn(x_1, x_2, w, w_0, v, v_01, v_02):

    z_1 = x_1 * w[0][0] + x_2 * w[0][1] + w_0
    z_2 = x_1 * w[1][0] + x_2 * w[1][1] + w_0
    z_3 = x_1 * w[2][0] + x_2 * w[2][1] + w_0
    z_4 = x_1 * w[3][0] + x_2 * w[3][1] + w_0

    u_1 = relu(z_1) * v[0][0] + relu(z_2) * v[0][1] + \
        relu(z_3) * v[0][2] + relu(z_4) * v[0][3] + v_01
    u_2 = relu(z_1) * v[1][0] + relu(z_2) * v[1][1] + \
        relu(z_3) * v[1][2] + relu(z_4) * v[1][3] + v_02

    u_exp_sum = np.exp(relu(u_1)) + np.exp(relu(u_2))

    o_1 = np.exp(relu(u_1)) / u_exp_sum
    o_2 = np.exp(relu(u_2)) / u_exp_sum

    return (o_1, o_2)
