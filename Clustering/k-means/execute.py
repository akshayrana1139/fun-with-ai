import kmeans as kk
import pandas as pd

# Reading and processing data.
input_data = pd.read_csv('exzeo_data.csv')
train_data = input_data.copy()
from sklearn import preprocessing
headers = list(train_data.columns.values)
for i in headers:
    train_data[i] = preprocessing.LabelEncoder().fit(train_data[i]).transform(train_data[i])


# Evaluating correct value for n_cluster. Uncomment the below part to see results.

# from sklearn.metrics import silhouette_score
# for n_cluster in range(2, 8):
#     # Trying kmeans algorithm with different k values to evaluate Silhouette Coefficient.
#     centroids, output_labels = kk.kmeans(train_data,n_cluster)
#     sil_coeff = silhouette_score(train_data, output_labels, metric='euclidean')
#     print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

################## PRINTED VALUES ####################################
# For n_clusters=2, The Silhouette Coefficient is 0.29845177144168505
# For n_clusters=3, The Silhouette Coefficient is 0.30709875981841545
# For n_clusters=4, The Silhouette Coefficient is 0.3030068577407267
# For n_clusters=5, The Silhouette Coefficient is 0.24108270808451635
# For n_clusters=6, The Silhouette Coefficient is 0.24655759914862482
# For n_clusters=7, The Silhouette Coefficient is 0.2297862596105836
# For n_clusters=8, The Silhouette Coefficient is 0.20669274487306408
################## PRINTED VALUES ####################################

# Choosing k = 3 as the Silhoute coefficient is highest for it.
centroids, output_labels = kk.kmeans(train_data,3)
input_data["clusters"] = output_labels
input_data.to_csv("exzeo_data_output.csv", sep=',' ,index=False )
print("The output file is generated.")

# Reducing dimensions of the Feature matrix to plot 2D using PCA [Works in Jupyter]
import pylab as pl
from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(train_data)
pca_2d = pca.transform(train_data)
pl.figure('K-means with 3 clusters')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=output_labels)
pl.show()


# Below lines to plot 3Dgraph. Takes time and saves an html file. Need patience for it to open.

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_array = pca.fit_transform(train_data)
data_to_plot = pd.DataFrame({'X1':pca_array[:,0],'X2':pca_array[:,1],'X3':pca_array[:,2], 'output':output_labels[:,0]})
kk.plot3d(data_to_plot)
