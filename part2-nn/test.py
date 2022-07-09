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

    print("O_1: ", o_1)
    print("O_2: ", o_2)


def main():
    try:
        # check_basic_relu()
        # check_logic_nand()
        # check_pseudo_and()
        check_simple_nn()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
