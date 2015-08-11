import matplotlib
from matplotlib import rcsetup
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

print plt.matplotlib.matplotlib_fname()
print(rcsetup.all_backends)
__author__ = 'laurogama'


def train_iris_dataset():
    iris = load_iris()

    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # log_reg = LogisticRegression()
    k_range = range(1, 26)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # print k
        scores.append(classifier_score(knn, X_train, y_train, X_test, y_test))
        # classifier_score(log_reg, X_train, y_train, X_test, y_test)
    print scores
    plt.plot(k_range, scores)
    plt.xlabel("value of K for Knn")
    plt.ylabel("Testing accuracy")
    plt.show()


def classifier_score(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_score = metrics.accuracy_score(y_test, y_pred)
    # print  "{} accuracy_score: {}".format(classifier.__class__, y_score)
    return y_score


if __name__ == "__main__":
    train_iris_dataset()
