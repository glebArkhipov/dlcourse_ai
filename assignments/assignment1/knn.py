import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided

        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = self.get_manhattan_dist(X[i_test], self.train_X[i_train])
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = np.absolute(X[i_test] - self.train_X).sum(1)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        # Using float32 to save memory - the default is float64
        dists = np.absolute(X[:, None, :] - self.train_X[None, :, :], dtype=np.float32).sum(-1)
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            pred[i] = self.get_closest_k(dists[i])
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case

        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            pass
        return pred

    @staticmethod
    def get_manhattan_dist(array_1, array_2):
        return np.absolute(array_1 - array_2).sum()

    def get_closest_k(self, dists):
        """
        :param dists: array - dists from one point
        :return: prediction for one point
        """
        points_and_dists = list(enumerate(dists))
        closest_points_and_dists = sorted(points_and_dists[:self.k], key=lambda tup: tup[1])
        for index, dist in points_and_dists[self.k:]:
            farthest_of_closest = closest_points_and_dists[self.k - 1]
            if farthest_of_closest[1] > dist:
                closest_points_and_dists[self.k - 1] = (index, dist)
            closest_points_and_dists = sorted(closest_points_and_dists, key=lambda tup: tup[1])
        results = []
        for (index, _) in closest_points_and_dists:
            results.append(self.train_y[index])
        return self.get_the_most_common_elements(results)

    @staticmethod
    def get_the_most_common_elements(arrays):
        elements_and_counts = [(el, arrays.count(el)) for el in set(arrays)]
        return max(elements_and_counts, key=lambda el_counts: el_counts[1])[0]
