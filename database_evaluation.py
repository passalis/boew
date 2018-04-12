from sklearn.neighbors import NearestNeighbors
import numpy as np
import cPickle


class Database(object):
    def __init__(self, database_vectors, targets, metric='euclidean'):

        self.nn = NearestNeighbors(n_neighbors=database_vectors.shape[0], algorithm='brute', metric=metric)
        self.nn.fit(database_vectors)
        self.targets = targets
        bins = np.bincount(targets)
        idx = np.nonzero(bins)[0]
        self.instances_per_target = dict(zip(idx, bins[idx]))
        self.number_of_instances = float(len(targets))
        self.recall_levels = np.arange(0, 1.01, 0.1)
        self.fine_recall_levels = np.arange(0, 1.01, 0.05)

    def get_binary_relevances(self, queries, targets):
        """
        Executes the queries and returns the binary relevance vectors (one vector for each query)
        :param queries: the queries
        :param targets: the label of each query
        :return:
        """
        distances, indices = self.nn.kneighbors(queries)
        relevant_vectors = np.zeros_like(indices)
        for i in range(targets.shape[0]):
            relevant_vectors[i, :] = self.targets[indices[i, :]] == targets[i]
        return relevant_vectors

    def get_metrics(self, relevant_vectors, targets):
        """
        Evaluates the retrieval performance
        :param relevant_vectors: the relevant vectors for each query
        :param targets: labels of the queries
        :return:
        """
        # Speedup trick
        dtype = 'float32'
        relevant_vectors = np.asarray(relevant_vectors, dtype=dtype)
        targets = np.asarray(targets, dtype=dtype)

        # Calculate precisions per query
        precision = np.cumsum(relevant_vectors, axis=1) / np.arange(1, self.number_of_instances + 1)
        precision = np.asarray(precision, dtype=dtype)
        # Calculate recall per query
        instances_per_query = np.zeros((targets.shape[0], 1), dtype=dtype)
        for i in range(targets.shape[0]):
            instances_per_query[i] = self.instances_per_target[targets[i]]
        recall = np.cumsum(relevant_vectors, axis=1) / instances_per_query

        # Calculate interpolated precision
        interpolated_precision = np.zeros_like(precision)
        for i in range(precision.shape[1]):
            interpolated_precision[:, i] = np.max(precision[:, i:], axis=1)

        # Calculate precision @ 11 recall point
        precision_at_recall_levels = np.zeros((targets.shape[0], self.recall_levels.shape[0]), dtype=dtype)
        for i in range(len(self.recall_levels)):
            idx = np.argmin(np.abs(recall - self.recall_levels[i]), axis=1)
            precision_at_recall_levels[:, i] = interpolated_precision[np.arange(targets.shape[0]), idx]

        # Calculate fine-grained precision
        precision_at_fine_recall_levels = np.zeros((targets.shape[0], self.fine_recall_levels.shape[0]), dtype=dtype)
        for i in range(len(self.fine_recall_levels)):
            idx = np.argmin(np.abs(recall - self.fine_recall_levels[i]), axis=1)
            precision_at_fine_recall_levels[:, i] = interpolated_precision[np.arange(targets.shape[0]), idx]

        # Calculate the means values of the metrics
        ap = np.mean(precision_at_recall_levels, axis=1)
        m_ap = np.mean(ap)
        interpolated_precision = np.mean(interpolated_precision, axis=0)
        interpolated_fine_precision = np.mean(precision_at_fine_recall_levels, axis=0)

        return m_ap, interpolated_precision, self.fine_recall_levels, interpolated_fine_precision

    def evaluate(self, queries, targets):
        """
        Evaluates the performance of the database using the following metrics: interpolated map, interpolated precision,
        and precision-recall curve
        :param queries: the queries
        :param targets: the labels
        :return: the evaluated metrics
        """
        relevant_vectors = self.get_binary_relevances(queries, targets)
        (map, interpolated_precision, self.fine_recall_levels, interpolated_fine_precision) = \
            self.get_metrics(relevant_vectors, targets)
        res = DatabaseResults(map, interpolated_precision, self.fine_recall_levels, interpolated_fine_precision)
        return res


class DatabaseResults:
    def __init__(self, map=0, i_precision=0, f_recall=0, f_i_precision=0):
        if map != 0:
            self.map = [map]
            self.interpolated_precision = [i_precision]
            self.fine_iterpolated_precision = [f_i_precision]
            self.fine_recall_levels = f_recall
        else:
            self.map = []
            self.interpolated_precision = []
            self.fine_iterpolated_precision = []
            self.fine_recall_levels = []

        self.recall_levels = np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    def get_metrics(self):
        """
        Returns the mean and the std of the stored metrics
        :return:
        """
        map = np.asarray(self.map)

        interpolated_precision = np.asarray(self.interpolated_precision)
        fine_iterpolated_precision = np.asarray(self.fine_iterpolated_precision)

        map_mean = np.mean(map, axis=0)
        interpolated_precision_mean = np.mean(interpolated_precision, axis=0)
        fine_iterpolated_precision_mean = np.mean(fine_iterpolated_precision, axis=0)

        map_std = np.std(map, axis=0)
        interpolated_precision_std = np.std(interpolated_precision, axis=0)
        fine_iterpolated_precision_std = np.std(fine_iterpolated_precision, axis=0)

        results = {'map_mean': map_mean, 'map_std': map_std, 'interpolated_precision_mean': interpolated_precision_mean,
                   'interpolated_precision_std': interpolated_precision_std, 'fine_interpolated_precision_mean':
                       fine_iterpolated_precision_mean,
                   'fine_iterpolated_precision_std': fine_iterpolated_precision_std,
                   'recall_levels': self.recall_levels}

        return results
