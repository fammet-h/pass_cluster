# https://www.dskomei.com/entry/2018/04/03/004543
# http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf

import numpy as np
from scipy import stats, linalg
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import math


class KMedoids(object):

    def __init__(self, n_cluster, max_iter=300, n_init=10):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.n_init = n_init

    def fit_predict(self, D):
        m, _ = D.shape

        col_names = ['x_' + str(i + 1) for i in range(m)]

        best_results = None
        best_sse = np.Inf
        best_medoids = None
        for init_num in range(self.n_init):
            initial_medoids = np.random.choice(range(m), self.n_cluster, replace=False)
            tmp_D = D[:, initial_medoids]

            # Clustering to the closest centroid in the initial centroids
            labels = np.argmin(tmp_D, axis=1)

            # Create a dataframe that has an ID 
            # because it is easier to handle when a unique ID is assigned to each point
            results = pd.DataFrame([range(m), labels]).T
            results.columns = ['id', 'label']

            # The ID of each point is linked to the distance matrix
            # The columns of the distance matrix are named so that they can be easily processed later
            results = pd.concat([results, pd.DataFrame(D, columns=col_names)], axis=1)

            before_medoids = initial_medoids
            new_medoids = []

            loop = 0
            
            # There is a change in the group of medoids, 
            # and continue if the number of loops is less than max_iter
            while len(set(before_medoids).intersection(set(new_medoids))) != self.n_cluster and loop < self.max_iter:

                if loop > 0:
                    before_medoids = new_medoids.copy()
                    new_medoids = []

                # In each cluster, the point with the smallest total distance from other points in the cluster
                # is the new cluster
                for i in range(self.n_cluster):
                    tmp = results.loc[results['label'] == i, :].copy()

                    # At each point, the total distance from other points is calculated
                    tmp['distance'] = np.sum(tmp.loc[:, ['x_' + str(id + 1) for id in tmp['id']]].values, axis=1)
                    tmp = tmp.reset_index(drop=True)
                    
                    # The distance calculated above makes the first point a new medoid
                    new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

                new_medoids = sorted(new_medoids)
                tmp_D = D[:, new_medoids]

                # Newly select the cluster with the smallest distance in the new medoid
                clustaling_labels = np.argmin(tmp_D, axis=1)
                results['label'] = clustaling_labels

                loop += 1

            # Add required information to results
            results = results.loc[:, ['id', 'label']]
            results['flag_medoid'] = 0
            for medoid in new_medoids:
                results.loc[results['id'] == medoid, 'flag_medoid'] = 1
                
            # Distance to each medoid
            tmp_D = pd.DataFrame(tmp_D, columns=['medoid_distance'+str(i) for i in range(self.n_cluster)])
            results = pd.concat([results, tmp_D], axis=1)

            sse = self.sse(distances=D, predicted_values=results['label'].values, medoids=new_medoids)

            if sse < best_sse:
                best_sse = sse
                best_results = results.copy()
                best_medoids = new_medoids.copy()

        self.labels_ = best_results['label'].values
        self.results = best_results
        self.cluster_centers_ = np.array(best_medoids)
        self.inertia_ = best_sse

        return self.labels_


    def fit(self, D):

        m, _ = D.shape

        col_names = ['x_' + str(i + 1) for i in range(m)]

        best_results = None
        best_sse = np.Inf
        best_medoids = None
        for _ in range(self.n_init):

            initial_medoids = self.making_initial_medoids(D, n_cluster=self.n_cluster)
            tmp_D = D[:, initial_medoids]

            # Clustering to the closest centroid in the initial centroids
            labels = np.argmin(tmp_D, axis=1)
            results = pd.DataFrame([range(m), labels]).T
            results.columns = ['id', 'label']

            results = pd.concat([results, pd.DataFrame(D, columns=col_names)], axis=1)


            before_medoids = initial_medoids
            new_medoids = []

            loop = 0
            while len(set(before_medoids).intersection(set(new_medoids))) != self.n_cluster and loop < self.max_iter:

                if loop > 0:
                    before_medoids = new_medoids.copy()
                    new_medoids = []

                for i in range(self.n_cluster):
                    tmp = results.loc[results['label'] == i, :].copy()
                    tmp['distance'] = np.sum(tmp.loc[:, ['x_' + str(id + 1) for id in tmp['id']]].values, axis=1)
                    tmp.reset_index(inplace=True)
                    new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

                new_medoids = sorted(new_medoids)
                tmp_D = D[:, new_medoids]
                labels = np.argmin(tmp_D, axis=1)
                results['label'] = labels

                loop += 1

            results = results.loc[:, ['id', 'label']]
            results['flag_medoid'] = 0
            for medoid in new_medoids:
                results.loc[results['id'] == medoid, 'flag_medoid'] = 1
            tmp_D = pd.DataFrame(tmp_D, columns=['label' + str(i) + '_distance' for i in range(self.n_cluster)])
            results = pd.concat([results, tmp_D], axis=1)

            sse = self.sse(distances=D, predicted_values=results['label'].values, medoids=new_medoids)

            if sse < best_sse:
                best_sse = sse
                best_results = results.copy()
                best_medoids = new_medoids.copy()

        self.results = best_results
        self.cluster_centers_ = np.array(best_medoids)
        self.labels_ = self.results['label'].values

        return self


    def sse(self, distances, predicted_values, medoids):
        """
        calculate sse (sum of squared errors of prediction)
        """
        unique_labels = sorted(np.unique(predicted_values))

        sse = []
        for label, medoid in zip(unique_labels, medoids):
            
            # Distance from center of each cluster
            distance = distances[medoid, predicted_values == label]
            distance_squared = distance * distance
            sse.extend(distance_squared.tolist())
        return np.sum(sse)
    
    
    def making_initial_medoids(self, distances, n_cluster):
        """
        making initial medoids
        """
        m, n = distances.shape

        distances_pd = pd.DataFrame({'id':range(m)})
        distances_pd = pd.concat([distances_pd, pd.DataFrame(distances, columns=[i for i in range(n)])], axis=1)

        medoids = []
        for cluster_num in range(n_cluster):

            if cluster_num == 0:
                medoid = np.random.randint(0, m, size=1)
                medoids.extend(medoid)
            else:
                distance = distances_pd.drop(medoids, axis=0)
                distance = distance.loc[:, ['id'] + medoids]
                
                # Seeking the nearest center
                distance['min_distance'] = distance.min(axis=1)
                distance['min_distance_squared'] = distance['min_distance']*distance['min_distance']
                ids = distance['id'].values
                
                # Seeking the probability distribution
                distance_values = distance['min_distance_squared'] / np.sum(distance['min_distance_squared'])
                medoid = ids[np.random.choice(range(ids.size), 1, p=distance_values)]
                medoids.extend(medoid)

        medoids = sorted(medoids)
        return medoids


class XMedoids:


    def __init__(self, n_cluster=2, max_iter=9999, n_init=1):
        """
        k_init: The initial number of clusters applied to KMeans()
        """
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.n_init = n_init


    def fit(self, X):
        """
        X: array-like or sparse matrix, shape=(n_samples, n_features)
        """
        self.__clusters = []
        
        # param n_init=self.n_init needed in k_medoids
        clusters = self.Cluster.build(X, KMedoids(n_cluster=self.n_cluster, max_iter=self.max_iter, n_init=self.n_init).fit(squareform(pdist(X))))
        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype=np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self


    def __recursively_split(self, clusters):
        """
        clusters: list-like object, which contains instances of 'XMedoids.Cluster'
        """
        for cluster in clusters:
            if cluster.size <= cluster.data.shape[1] * 2:
                self.__clusters.append(cluster)
                continue

            k_medoids = KMedoids(2, max_iter=self.max_iter, n_init=self.n_init).fit(squareform(pdist(cluster.data)))
            c1, c2 = self.Cluster.build(cluster.data, k_medoids, cluster.index)

            if np.linalg.det(c1.cov) == 0 and np.linalg.det(c2.cov) == 0:
                beta = 0
            else:
                beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            alpha = 0.5 / stats.norm.cdf(beta)
            bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)

            if bic < cluster.bic():
                self.__recursively_split([c1, c2])
            else:
                self.__clusters.append(cluster)


    class Cluster:

        @classmethod
        def build(cls, X, cluster_model, index=None):

            if index is None:
                index = np.array(range(0, X.shape[0]))
            labels = range(0, len(np.unique(cluster_model.labels_)))

            return tuple(cls(X, index, cluster_model, label) for label in labels)


        def __init__(self, X, index, cluster_model, label):
            """
            index: vector showing which row of the original data the X's sample of each row corresponds to
            """
            self.data = X[cluster_model.labels_ == label]
            self.index = index[cluster_model.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            center_ = cluster_model.cluster_centers_[label]
            self.center = X.values[center_]
            # self.center = X[center_, :]
            self.cov = np.cov(self.data.T)


        def log_likelihood(self):
            # return sum([stats.multivariate_normal.logpdf(x, self.center, self.cov) for x in self.data])
            x_ = []
            for _, x in self.data.iterrows():
                x_i = []
                for i in range(self.data.shape[1]):
                    x_i.append(x[i])
                x_.append(x_i)
            # return math.log(sum([stats.multivariate_normal.pdf(x, self.center, self.cov) for x in x_]))
            try:
                log_likelihood = sum([stats.multivariate_normal.logpdf(x, self.center, self.cov) for x in x_])
            except linalg.LinAlgError:
                log_likelihood = 0
            except ValueError:
                log_likelihood = 0
            return log_likelihood


        def bic(self):
            """
            Bayesian Information Criterion
            """
            return -2 * self.log_likelihood() + self.df * np.log(self.size)