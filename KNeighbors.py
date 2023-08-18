import time
from scipy.ndimage import shift
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl

def Download_MINST():
    mnist = fetch_openml('mnist_784', parser="auto")
    print("Data Keys: ",mnist.keys())
    print("Data Download Completed!")
    return mnist

def Divide_Data(mnist):
    print("Dividing Data!")
    x, y = mnist["data"], mnist["target"]
    #print(x.shape)
    #print(y.shape)

    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    return X_train, X_test, y_train, y_test

def Shift_Images(X_train, y_train):
    begin = time.time()
    trainX_aug = []
    trainY_aug = []

    for image, label in zip(np.array(X_train), y_train):
        # shift down
        shift_image = shift(image.reshape((28, 28)), [1, 0], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.append(label)
        # shift up
        shift_image = shift(image.reshape((28, 28)), [-1, 0], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.append(label)

        # shift right
        shift_image = shift(image.reshape((28, 28)), [0, 1], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.append(label)
        # shift left
        shift_image = shift(image.reshape((28, 28)), [0, -1], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.append(label)

    trainX_aug = pd.concat([trainX_aug, pd.DataFrame([X_train])], ignore_index=True)
    trainY_aug = pd.Series(trainY_aug).append(y_train, ignore_index=True)
    end = time.time()
    print("Time to run:", end - begin)
    return trainX_aug, trainY_aug

def Saving_Data(X_train, X_test, y_train, y_test):
    print("Saving Data!")
    X_train.to_csv("X_train", sep=',', index=False, encoding='utf-8')
    X_test.to_csv("X_test", sep=',', index=False, encoding='utf-8')
    y_train.to_csv("y_train", sep=',', index=False, encoding='utf-8')
    y_test.to_csv("y_test", sep=',', index=False, encoding='utf-8')
    print("Saving Completed!")

def Loading_Data():
    print("Loading Data!")
    X_train = pd.read_csv("Data/X_train", sep=',')
    X_test = pd.read_csv("Data/X_test", sep=',')
    y_train = pd.read_csv("Data/y_train", sep=',')
    y_test = pd.read_csv("Data/y_test", sep=',')
    print("Loading Completed!")

    return X_train, X_test, y_train, y_test

def K_Neighbors_Classifire(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    print("Training Completed!")

    """
    Test:
     
    for a in range(1, 100000):
        if y_test[a-1:a].values != knn.predict(X_test[a-1:a]):
            print("Error: ", "Test Case = ", a, "Test Answer = ",y_test[a-1:a].values,"Predicted Asnwer = ", knn.predict(X_test[a-1:a]))
    print("Test Completed!")
    """

    return knn

def Test_Model(knn, X_test, y_test):
    print("Testing Model!")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def Save_Model(knn, filename):
    filename += '.pkl'
    joblib.dump(knn, filename)
    print("Saved Model: ", filename)

def Load_Model(filename):
    filename += '.pkl'
    print("Loading Model: ", filename)
    return joblib.load(filename)

if __name__ == "__main__":

    #Downloading and saving mnist
    #
    #mnist = Download_MINST()
    #X_train, X_test, y_train, y_test = Divide_Data(mnist)
    #Saving_Data(X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = Loading_Data()

    X_train_aug, y_train_aug = Shift_Images(X_train, y_train)

    #knn = K_Neighbors_Classifire(X_train, y_train)
    #Test_Model(knn, X_test, y_test)
    #Save_Model(knn, 'knn_Library')
    #knn = Load_Model('knn_Library')
    #Test_Model(knn, X_test, y_test)
