import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def euclidian(a, b):
    return np.linalg.norm(a-b)


################################       ALGORITHM       #############################################
#
# Input : K, set of points x1...xn
# Place centroids c1..ck at random locations.
# Repeat until convergence
#    - for each point x(i):
#         --find nearest centroid c(j)
#         --assign the point x(i) to cluster j
#    - for each cluster j = 1 .. K:
#         --new centroid c(j) = mean of all points x(i) assigned to cluster j in previous step.
# Stop when none of the cluster changes
#
####################################################################################################
def kmeans(dataset, k):
    dataset = dataset.as_matrix()
    rows, features = dataset.shape
    prototypes = dataset[np.random.randint(0, rows - 1, size=k)]
    prototypes_old = np.zeros(prototypes.shape)
    labels = np.zeros((rows, 1))
    norm = euclidian(prototypes, prototypes_old)
    iteration = 0
    while norm > 0:
        iteration += 1
        norm = euclidian(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = euclidian(prototype,
                                                        instance)

            labels[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(labels)) if labels[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes
    return prototypes, labels



# 3D Plot after reducing dimension to 3 using PCA
def plot3d(data_to_plot):

    from sklearn.datasets import make_blobs
    from sklearn.datasets import make_gaussian_quantiles
    from sklearn.datasets import make_classification, make_regression
    from sklearn.externals import six
    import pandas as pd
    import numpy as np
    import argparse
    import json
    import re
    import os
    import sys
    import plotly
    plotly.offline.init_notebook_mode()
    import plotly.graph_objs as go
    

    cluster1=data_to_plot.loc[data_to_plot['output'] == 0.0]
    cluster2=data_to_plot.loc[data_to_plot['output'] == 1.0]
    cluster3=data_to_plot.loc[data_to_plot['output'] == 2.0]
    scatter1 = dict(
        mode = "markers",
        name = "Cluster 1",
        type = "scatter3d",    
        x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
        marker = dict( size=2, color='green')
    )
    scatter2 = dict(
        mode = "markers",
        name = "Cluster 2",
        type = "scatter3d",    
        x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
        marker = dict( size=2, color='blue')
    )
    scatter3 = dict(
        mode = "markers",
        name = "Cluster 3",
        type = "scatter3d",    
        x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
        marker = dict( size=2, color='red')
    )
    cluster1 = dict(
        alphahull = 5,
        name = "Cluster 1",
        opacity = .1,
        type = "mesh3d",    
        x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
        color='green', showscale = True
    )
    cluster2 = dict(
        alphahull = 5,
        name = "Cluster 2",
        opacity = .1,
        type = "mesh3d",    
        x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
        color='blue', showscale = True
    )
    cluster3 = dict(
        alphahull = 5,
        name = "Cluster 3",
        opacity = .1,
        type = "mesh3d",    
        x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
        color='red', showscale = True
    )
    layout = dict(
        title = 'Interactive Cluster Shapes in 3D',
        scene = dict(
            xaxis = dict( zeroline=True ),
            yaxis = dict( zeroline=True ),
            zaxis = dict( zeroline=True ),
        )
    )
    fig = dict( data=[scatter1, scatter2, scatter3, cluster1, cluster2, cluster3], layout=layout )
    plotly.offline.plot(fig, filename='3D_cluster.html')
