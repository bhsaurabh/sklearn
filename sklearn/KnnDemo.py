#!/usr/bin/python -tt

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


class KnnDemo(object):
    @staticmethod
    def split_data(x, y):
        """ Splits data (features and target)
        Splits the various data into training and test data sets

        Args:
            X: A mxn matrix having m examples, each having n features
            y: A mx1 vector having target variables

        Returns:
            X_train: example features to be used for training
            y_train: target variables to be used for training
            X_test: example features to be used as test set
            y_test: target variables to be used as test set
        """
        # Seed the random generator
        np.random.seed(0)
        indices = np.random.permutation(len(x))

        # Use 70 % as training set
        limit = int(0.7 * len(x))

        # Create training set and test set
        x_train = x[indices[:limit]]
        y_train = y[indices[:limit]]
        x_test = x[indices[limit:]]
        y_test = y[indices[limit:]]

        return x_train, y_train, x_test, y_test

    def run(self):
        """ Application entry point

            Args:
                None
            Returns:
                None
        """
        # Load data sets
        iris = datasets.load_iris()
        iris_x, iris_y = iris.data, iris.target

        iris_x_train, iris_y_train, iris_x_test, iris_y_test = self.split_data(iris_x, iris_y)

        # Create and train a K-nearest neighbors classifier
        knn = KNeighborsClassifier()
        knn.fit(iris_x_train, iris_y_train)

        # Make predictions
        y_predicted = knn.predict(iris_x_test)

        # Print predictions
        for i in range(len(iris_x_test)):
            print('Predicted class: %d, Actual class: %d' % (y_predicted[i], iris_y_test[i]))

        # Calculate score
        score = knn.score(iris_x_test, iris_y_test)
        print('\nScore of prediction: ' + str(score))


# Standard boiler-plate notification
if __name__ == '__main__':
    knnDemo = KnnDemo()
    knnDemo.run()