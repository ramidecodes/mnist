import numpy as np
import traceback
import unit8_exe


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


def check_logic_nand():
    x_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    nand_truth = [1, 1, 1, 0]

    y_list = []
    for x in x_list:
        y = unit8_exe.logic_nand(x)
        y_list.append(y)
        log("Y: ", y)

    if y_list == nand_truth:
        log(green("PASS"), "logic_nand")
    else:
        log(red("FAILED"), "Import mnist project")


def main():
    # log(green("PASS"), "Import mnist project")
    try:
        # check_basic_relu()
        check_logic_nand()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
