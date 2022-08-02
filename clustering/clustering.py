from sklearn_extra.cluster import KMedoids
import numpy as np

# Using L2 Metric


def kmedoids(X):
    kmedoids = KMedoids(n_clusters=2,
                        metric="euclidean",
                        random_state=0,
                        method="pam").fit(X)
    labels = kmedoids.labels_
    cluster_centers = kmedoids.cluster_centers_

    return (labels, cluster_centers)
