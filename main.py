from sklearn.datasets import fetch_openml, load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

def Download_MINST():
    mnist = fetch_openml('mnist_784', parser="auto")
    print(mnist.keys())
    return mnist

def test_mnist(mnist):
    x, y = mnist["data"], mnist["target"]
    #print(x.shape)
    #print(y.shape)

    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = x[:60000],x[60000:], y[:60000], y[60000:]

    #simple test case (5 only)
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print(confusion_matrix(y_train_5, y_train_pred))


    #See the image

    #print(x[3:4].values.reshape((28,28)))
    #test_digit = x[:1]
    #test_digit_image = test_digit.values.reshape((28,28))
    #plt.imshow(test_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
    #plt.axis("off")
    #plt.show()

if __name__ == "__main__":
    mnist = Download_MINST()
    test_mnist(mnist)