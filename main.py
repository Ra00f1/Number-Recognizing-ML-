from sklearn.datasets import fetch_openml, load_digits
import pandas as pd
import matplotlib.pyplot as plt

def Download_MINST():
    mnist = fetch_openml('mnist_784')
    print(mnist.keys())
    return mnist

def test_mnist(mnist):
    x, y = mnist["data"], mnist["target"]
    print(x.shape)
    print(y.shape)

if __name__ == "__main__":
    mnist = Download_MINST()
    test_mnist(mnist)