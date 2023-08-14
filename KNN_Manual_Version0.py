import  pandas as pd
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl

def Euclidean_Distance(data):
    data = data[:1000]
    print("Calculating Euclidean Distance!")
    # Convert the data frame to a NumPy array
    array = data.to_numpy()
    # Assuming 'array' is a NumPy array with shape (length, num_features)
    length = array.shape[0]

    # Initialize the lists to store distances and indices
    train_distance_list = []
    train_id_counter = []
    start = time.time()

    df = pd.DataFrame(columns=['index', 'distance'])

    for i in range(length):
        if i%100 == 0:
            end = time.time()
            print(i,"time: ", end - start)
            start = time.time()

        # Calculate the Euclidean distance between array[i] and all other elements in 'array'
        distance = np.sqrt(np.sum((array[i] - array) ** 2, axis=1))
        train_distance_list.extend(distance)
        train_id_counter.extend([i] * length)

    d = {'index': train_id_counter, 'distance': train_distance_list}
    df = pd.DataFrame(d, columns=['index', 'distance'])
    df_sorted = df.sort_values(['index', 'distance'])

    print(df_sorted)
    return df_sorted

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

    distance_df = Euclidean_Distance(X_train)

    # Save Data Frame
    #distance_df.to_csv("Distance_DF", sep=',', index=False, encoding='utf-8')

    # Load Distance Data Frame
    #distance_df = pd.read_csv("Distance_DF", sep=',')
    #
    for i in range(10):
        test_digit = X_train[distance_df[i:i+1].index.values[0]: distance_df[i:i+1].index.values[0]+1]
        test_digit_image = test_digit.values.reshape((28, 28))
        plt.imshow(test_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
        plt.axis("off")
        plt.show()

    #counter = Counter(y_train[distance_df.index])
