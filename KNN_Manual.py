import  pandas as pd

def Euclidean_Distance(data):
    distance = 0
    for i in range(28):
        distance += (data)

def Loading_Data():
    print("Loading Data!")
    X_train = pd.read_csv("X_train", sep=',')
    X_test = pd.read_csv("X_test", sep=',')
    y_train = pd.read_csv("y_train", sep=',')
    y_test = pd.read_csv("y_test", sep=',')
    print("Loading Completed!")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = Loading_Data()

    for a in range(len(X_train)):
        data_raw = X_train[a:a+1]
        data = data_raw.values.reshape((28,28))
        Euclidean_Distance(data)
        break

