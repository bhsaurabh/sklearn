#!/usr/bin/python -tt

from sklearn import datasets
from sklearn.linear_model import LinearRegression


class LinearRegressionDemo(object):

    @staticmethod
    def load_diabetes_data():
        """
        Load the diabetes data set from scikit learn

        Args:
            None

        Returns:
            diabetes_X_train: Training features for diabetes data set
            diabetes_X_test: Test set features for diabetes data set
            diabetes_y_train: Target variables of the training set
            diabetes_y_test: Target variables of the test set
        """
        diabetes = datasets.load_diabetes()
        diabetes_X, diabetes_y = diabetes.data, diabetes.target

        # Split the data set as
        # 70 % -> Training set
        # 30 % -> Test set

        limit = 0.7 * len(diabetes_y)
        diabetes_X_train = diabetes_X[:limit]
        diabetes_X_test = diabetes_X[limit:]
        diabetes_y_train = diabetes_y[:limit]
        diabetes_y_test = diabetes_y[limit:]
        return diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test

    @staticmethod
    def train_estimator(X_train, y_train):
        """
        Creates and trains a linear regression estimator

        Args:
            X_train: mxn matrix having m training examples, each with n features
            y_train: mx1 vector having training target features

        Returns:
            regr: A linear regression estimator that has been fitted to the training data
        """
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        return regr

    def estimation_accuracy(self, regr, X_test, y_test):
        """
        Measures the accuracy of the estimator on a test set

        Args:
            regr: The regression estimator to be tested
            X_test: Test set features (mxn matrix)
            y_test: Target variables for test set (mx1 vector)

        Returns:
            score: The score of the estimator
        """
        y_predicted = regr.predict(X_test)
        for i in range(len(y_test)):
            print('Predicted value: %f, Actual value: %f' % (y_predicted[i], y_test[i]))

        return regr.score(X_test, y_test)

    def main(self):
        """
        Application entry point
        """
        diabetes = self.load_diabetes_data()
        diabetes_X_train, diabetes_X_test = diabetes[0], diabetes[1]
        diabetes_y_train, diabetes_y_test = diabetes[2], diabetes[3]

        # Create and train a linear regression estimator
        regr = self.train_estimator(diabetes_X_train, diabetes_y_train)

        score = self.estimation_accuracy(regr, diabetes_X_test, diabetes_y_test)
        print('\nEstimator score: %f' % score)


# standard boiler plate to execute script
if __name__ == '__main__':
    regressionDemo = LinearRegressionDemo()
    regressionDemo.main()


