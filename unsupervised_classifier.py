import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import classifier as cl
import logging as log
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import euclidean_distances

print = log.info

def kmeans(x_train, x_test, y_train, y_test, classes=2, random_state=8, crossval=False):
    k_values = range(classes, 32, 2)
    k_best = 10
    cl.name = "kmeans_{}".format(cl.name)

    if (crossval):

        inertia_values = []

        for k in k_values:
            # Initializing KMeans
            KM = KMeans(n_clusters=k)

            # fitting with inputs and
            # Predicting the clusters
            y_kmeans = KM.fit_predict(x_train)

            inertia_values.append(KM.inertia_)

        plt.figure()
        plt.plot(k_values, inertia_values, 'o-')
        plt.title('inertia_{}'.format(cl.name))
        plt.ylabel('Sum of squared distances')
        plt.xlabel('K')
        plt.savefig("./result/{}_inertia".format(cl.name), dpi=1200)


    else:

        KM = KMeans(n_clusters=k_best)
        y_kmeans = KM.fit_predict(x_test)  # clusters

        predictions = []
        accuracy = 0
        for i in range(k_best):

            freq = np.unique(y_test[np.where(y_kmeans == i)], return_counts=True)
            indice = np.where(freq[1] == np.max(freq[1]))
            if len(indice[0]) > 1:
                indice = np.random.randint(low=0, high=len(indice[0]))
            predictions = np.ones(len(y_test[np.where(y_kmeans == i)])) * freq[0][indice]
            accuracy = accuracy + np.sum(predictions == y_test[np.where(y_kmeans == i)])

        plt.figure()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_kmeans, s=50, cmap='rainbow')
        centers = KM.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=20, alpha=0.5);
        plt.title('Cluster Kmeans')
        plt.ylabel('component 1')
        plt.xlabel('component 0')
        plt.savefig("./result/{}_kmeans_components".format(cl.name), dpi=1200)

        plt.figure()
        plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='#ff7f0e')
        plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], c='#1f77b4')
        if classes == 3:
            plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], c='red')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=20, alpha=0.5);
        plt.title('Dataset with labels'.format(cl.name))
        plt.ylabel('component 1')
        plt.xlabel('component 0')
        plt.savefig("./result/{}_kmeans_label".format(cl.name), dpi=1200)

        accuracy = accuracy / len(y_kmeans)
        
        '''
        # figure con le due componenti fisher:
        x_train, x_test = pr.fisher_function(x_train, x_test, y_train, y_test, random_state, n=2, best=True)
        plt.figure()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_kmeans, s=50, cmap='rainbow')
        centers = KM.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=20, alpha=0.5);
        plt.title('Cluster Kmeans on Fisher components')
        plt.ylabel('component 1')
        plt.xlabel('component 0')
        plt.savefig("./result/{}_kmeans_fisher".format(cl.name), dpi=1200)

        # import pdb; pdb.set_trace()

        plt.figure()
        plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='#ff7f0e')
        plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], c='#1f77b4')
        plt.title('Dataset with labels on Fisher components')
        plt.ylabel('component 1')
        plt.xlabel('component 0')
        plt.savefig("./result/{}_kmeans_labe_fisher".format(cl.name), dpi=1200)
        '''
        print("Accuracy: {}".format(accuracy))


def hierarchical(x_train, x_test, y_train, y_test, classes=2, random_state=8, crossval=False):
    k_values = range(classes, 50, 1)
    inertia_values = []
    k_best = 10
    cl.name = "hierarchical_{}".format(cl.name)

    if crossval:

        for k in k_values:
            print("evaluating with k value equals to = {}".format(k))
            AC = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')

            # fitting with inputs and
            # Predicting the clusters
            y_hierarc = AC.fit_predict(x_train)

            inertia_values.append(compute_Intertia(x_train, y_hierarc, k))

        plt.figure()
        plt.plot(k_values, inertia_values, 'o-')
        plt.title('inertia_{}'.format(cl.name))
        plt.ylabel('Sum of squared distances')
        plt.xlabel('K')
        plt.savefig("./result/{}_inertia".format(cl.name), dpi=1200)

    else:

        AC = AgglomerativeClustering(n_clusters=k_best, affinity='euclidean', linkage='ward')

        # fitting with inputs and
        # Predicting the clusters
        y_hierarc = AC.fit_predict(x_test)

        # plot dendrogram
        linked = linkage(x_test, 'ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.savefig("./result/{}_dendrogram_hierarchical".format(cl.name), dpi=1200)

        plt.figure()
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hierarc, s=10, cmap='rainbow')
        plt.title('components_{}'.format(cl.name))
        plt.ylabel('component 1')
        plt.xlabel('component 0')
        plt.savefig("./result/{}_hierarc_components".format(cl.name), dpi=1200)
        
        plt.figure()
        
        plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], c='#ff7f0e')
        plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], c='#1f77b4')
        if classes == 3:
            plt.scatter(x_test[y_test == 2, 0], x_test[y_test == 2, 1], c='red')
        plt.title('Dataset with labels'.format(cl.name))
        plt.ylabel('component 1')
        plt.xlabel('component 0')
        plt.savefig("./result/{}_hierarc_label".format(cl.name), dpi=1200)
        

        predictions = []
        accuracy = 0
        for i in range(k_best):

            freq = np.unique(y_test[np.where(y_hierarc == i)], return_counts=True)
            indice = np.where(freq[1] == np.max(freq[1]))

            if len(indice[0]) > 1:
                indice = np.random.randint(low=0, high=len(indice[0]))

            predictions = np.ones(len(y_test[np.where(y_hierarc == i)])) * freq[0][indice]
            accuracy = accuracy + np.sum(predictions == y_test[np.where(y_hierarc == i)])

        accuracy = accuracy / len(y_hierarc)
        print("Accuracy: {}".format(accuracy))


def compute_Intertia(x, y, k):
    inertia = 0
    for i in range(k):
        centroid = np.mean(x[np.where(y == i)], axis=0)
        centr_reshape = np.reshape(centroid, (1, centroid.shape[0]))
        # import pdb; pdb.set_trace()
        inertia = inertia + sum(euclidean_distances(x[np.where(y == i)], centr_reshape))

    return inertia
