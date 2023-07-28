from sklearn.datasets import fetch_openml, load_digits
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def Download_MINST():
    mnist = fetch_openml('mnist_784', parser="auto")
    print(mnist.keys())
    return mnist

def test_mnist(mnist):
    x, y = mnist["data"], mnist["target"]
    print(x.shape)
    print(y.shape)
    print(x[:1])
    image = mnist.data.to_numpy()
    plt.subplot(431)
    plt.imshow((image[0].reshape(28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.subplot(432)
    plt.imshow(image[1].reshape(28, 28), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.subplot(433)
    plt.imshow(image[3].reshape(28, 28), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.subplot(434)
    plt.imshow(image[4].reshape(28, 28), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.subplot(435)
    plt.imshow(image[5].reshape(28, 28), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.subplot(436)
    plt.imshow(image[6].reshape(28, 28), cmap=plt.cm.gray_r,
               interpolation='nearest')

if __name__ == "__main__":
    mnist = Download_MINST()
    test_mnist(mnist)