import clustering
import numpy as np
import traceback


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


def test_kmedoids():
    X = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])

    (labels, cluster_centers) = clustering.kmedoids(X)

    print("Labels: ", labels)
    print("Cluster Centers: ", cluster_centers)

    correct_centers = np.array([[0, 0], [0, -6]])
    correct_labels = np.array([1, 0, 0, 0])
    if ((cluster_centers == correct_centers).all() and (labels == correct_labels).all()):
        log(green("PASS"), "kmedoids")
    else:
        log(red("FAILED"), "kmedoids")


def main():
    try:
        test_kmedoids()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()
