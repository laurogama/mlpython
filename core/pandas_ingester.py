import matplotlib
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.base import LinearRegression

matplotlib.use('Qt4Agg')
import seaborn as sns
import numpy as np


def ingest_csv():
    data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',
                       index_col=0)
    feature_cols = ['TV', 'Radio']
    X = data[feature_cols]
    target_col = 'Sales'
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=1)
    sns.pairplot(data, x_vars=feature_cols, y_vars=target_col,
                 size=7, aspect=0.7, kind='reg')
    sns.plt.show()

    linear_reg = LinearRegression()
    classifier_root_mean_square_error(linear_reg, X_train, y_train, X_test,
                                      y_test)


def classifier_root_mean_square_error(classifier, X_train, y_train, X_test,
                                      y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_score = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print  "{} accuracy_score: {}".format(classifier.__class__, y_score)
    return y_score


if __name__ == '__main__':
    ingest_csv()
