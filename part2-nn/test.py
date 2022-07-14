from tkinter import X
from turtle import home
import numpy as np
import traceback
import matplotlib
import matplotlib.pyplot as plt
import unit8_exe
import homework_3


def green(s):
    return '\033[1;32m%s\033[m' % s


def yellow(s):
    return '\033[1;33m%s\033[m' % s


def red(s):
    return '\033[1;31m%s\033[m' % s


def log(*m):
    print(" ".join(map(str, m)))


def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def check_basic_relu():
    x = np.array([1, 0])
    w = np.array([[1], [-1]])
    w_0 = -3

    y = unit8_exe.basic_relu(x, w, w_0)

    log("Y: ", y)


def plot_scatter(coordinate_list, truth):
    # log("Coordinate List: ", coordinate_list)
    coordinate_matrix = np.array(coordinate_list)
    y, x = coordinate_matrix.T

    colors = ['red', 'blue']

    # plotting the points
    plt.scatter(x, y, c=truth, cmap=matplotlib.colors.ListedColormap(
        colors), alpha=0.4)
    # cmap=colors[truth]
    # naming the x axis
    plt.xlabel('x1 - axis')
    # naming the y axis
    plt.ylabel('x2 - axis')

    # giving a title to my graph
    plt.title('The X')

    # function to show the plot
    plt.show()


def check_logic_nand():
    x_list = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    nand_truth = np.array([1, 1, 1, 0])

    y_list = []
    for x in x_list:
        y = unit8_exe.logic_nand(x)
        y_list.append(y)
        log("Y: ", y)

    plot_scatter(x_list, nand_truth)

    if ((y_list == nand_truth).all()):
        log(green("PASS"), "logic_nand")
    else:
        log(red("FAILED"), "logic_nand")


def check_pseudo_and():
    x_list = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_truth = np.array([1, 0, 0, 1])

    y_list = []
    for x in x_list:
        y = unit8_exe.pseudo_and(x)
        y_list.append(y)
        log("Y: ", y)

    plot_scatter(y_list, and_truth)

    # if (y_list == and_truth):
    #     log(green("PASS"), "pseudo_and")
    # else:
    #     log("Y list:", y_list)
    #     log("nand truth: ", and_truth)
    #     log(red("FAILED"), "pseudo_and")


def check_simple_nn():
    x_1 = 3
    x_2 = 14

    w = np.array([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
    ])
    print("W shape: ", w.shape)

    w_0 = -1

    v = np.array([
        [1, 1, 1, 1],
        [-1, -1, -1, -1]
    ])
    v_01 = 0
    v_02 = 2

    print("V shape: ", v.shape)

    (o_1, o_2) = homework_3.simple_nn(x_1, x_2, w, w_0, v, v_01, v_02)

    o_1_result = 0.9999996940977731
    o_2_result = 3.059022269256247e-07

    print("O_1: ", o_1)
    print("O_2: ", o_2)

    if (o_1 == o_1_result and o_2_result):
        log(green("PASS"), "simple_nn")
    else:
        log(red("FAILED"), "simple_nn")


def check_simple_lstm():
    c = 0
    h = 0
    x_a = [0, 0, 1, 1, 1, 0]
    x_b = [1, 1, 0, 1, 1]
    w_fh = 0
    w_ih = 0
    w_oh = 0
    w_fx = 0
    w_ix = 100
    w_ox = 100
    w_ch = -100
    w_cx = 50
    b_f = -100
    b_i = 100
    b_o = 0
    b_c = 0

    h_list = []
    h_list_answer_a = [0, 0, 1, -1, 1, 0]
    h_list_answer_b = [1, -1, 0, 1, -1]

    for x_i in x_b:
        (c, h) = homework_3.simple_lstm(x_i, h, c, w_fh, w_fx,
                                        w_ih, w_ix, w_oh, w_ox, w_ch, w_cx, b_f, b_i, b_o, b_c)
        h_list.append(h)

    print("H List: ", h_list)

    if (h_list == h_list_answer_a or h_list == h_list_answer_b):
        log(green("PASS"), "simple_lstm")
    else:
        log(red("FAILED"), "simple_lstm")


def check_two_layer_nn():
    t = 1
    x = 3
    w_1 = 0.01
    w_2 = -5
    b = -1

    c = homework_3.two_layer_nn(t, x, w_1, w_2, b)

    print("C: ", c)


def check_simple_convolutional_nn():
    f = np.array([1, 3, -1, 1, -3])
    g = np.array([-1, 0, 1])

    (conv_valid, conv_same) = homework_3.simple_convolutional_nn(f, g)

    conv_no_padding = np.array([2, 2, 2])
    conv_padding = np.array([-3, 2, 2, 2, 1])

    print("Convolution Valid: ", conv_valid)  # No Padding
    print("Convolution No padding: ", conv_no_padding)
    print("Convolution Same: ", conv_same)  # With 0 Padding
    print("Convolution Padding: ", conv_padding)

    if (np.array_equal(conv_valid, conv_no_padding) and np.array_equal(conv_same, conv_padding)):
        log(green("PASS"), "simple_lstm")
    else:
        log(red("FAILED"), "simple_lstm")


def check_conv2d_sum():
    f = np.array([[1, 2, 1], [2, 1, 1], [1, 1, 1]])
    g = np.array([[1, 0.5], [0.5, 1]])

    answer = 15.

    (conv) = homework_3.conv2d(f, g)
    conv_sum = sum(sum(conv))

    print("Convolution Valid: ", conv)  # No Padding
    print("Convolution SUM: ", conv_sum)

    if (conv_sum == answer):
        log(green("PASS"), "simple_lstm")
    else:
        log(red("FAILED"), "simple_lstm")


def check_simple_cnn():
    i = np.array([[1, 0, 2], [3, 1, 0], [0, 0, 4]])
    f = np.array([[1, 0], [0, 1]])

    conv_result = homework_3.simple_cnn(i, f)
    answer = 5

    print("CNN Result: ", conv_result)

    if (conv_result == answer):
        log(green("PASS"), "simple_lstm")
    else:
        log(red("FAILED"), "simple_lstm")


def main():
    try:
        # check_basic_relu()
        # check_logic_nand()
        # check_pseudo_and()
        # check_simple_nn()
        # check_simple_lstm()
        # check_two_layer_nn()
        # check_simple_convolutional_nn()
        # check_conv2d_sum()
        check_simple_cnn()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
