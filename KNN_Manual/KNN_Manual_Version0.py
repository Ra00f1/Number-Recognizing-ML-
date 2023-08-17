import  pandas as pd
import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl

def Euclidean_Distance(data):
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
        if i%100 == 0 and i != 0:
            end = time.time()
            print(i,"time: ", end - start)
            d = {'index': train_id_counter , 'distance': train_distance_list}
            temp_df = pd.DataFrame(d, columns=['index', 'distance'])
            #temp_df = temp_df.sort_values(['index', 'distance'])
            df = pd.concat([df, temp_df], ignore_index=True)
            train_distance_list.clear()
            train_id_counter.clear()
            start = time.time()

        # Calculate the Euclidean distance between array[i] and all other elements in 'array'
        distance = np.sqrt(np.sum((array[i] - array) ** 2, axis=1))
        train_distance_list.extend(distance)
        train_id_counter.extend([i] * length)
        #print(df)

    #d = {'index': train_id_counter, 'distance': train_distance_list}
    #df = pd.DataFrame(d, columns=['index', 'distance'])
    df_sorted = df.sort_values(['index', 'distance'])
    #df_sorted = df
    print(df_sorted)
    return df_sorted

def Loading_Data():
    print("Loading Data!")
    X_train = pd.read_csv("../Data/X_train", sep=',')
    X_test = pd.read_csv("../Data/X_test", sep=',')
    y_train = pd.read_csv("../Data/y_train", sep=',')
    y_test = pd.read_csv("../Data/y_test", sep=',')
    print("Loading Completed!")

    return X_train, X_test, y_train, y_test

def minkowski_distance(length, a, b, p=2, ):
    train_distance_list = []
    train_id_counter = []
    distance = np.sqrt(np.sum((a - b) ** 2, axis=1))
    train_distance_list.extend(distance)
    train_id_counter.extend([length] * b.shape[0])
    d = {'index': train_id_counter, 'distance': train_distance_list}
    df = pd.DataFrame(d, columns=['index', 'distance'])
    df_sorted = df.sort_values(['index', 'distance'])
    return distance, df_sorted


def knn_predict(X_train, X_test, y_train, y_test, k, p):
    # Counter to help with label voting
    from collections import Counter

    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    X_train1 = X_train.to_numpy()
    X_test1 = X_test.to_numpy()
    y_train1 = y_train.to_numpy()
    y_test = y_test.to_numpy()
    length = 0

    for test_point in X_test1:
        distances = []
        distance, df_nn = minkowski_distance(length, test_point, X_train1, p=p)
        length += 1
        distances.append(distance)

        # Store distances in a dataframe
        #df_dists = pd.DataFrame(data=distances, columns=['dist'],
        #                        index=y_train.index)
#
        ## Sort distances, and only consider the k closest points
        #df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]

        # Append prediction to output list
        y_hat_test.append([prediction, test_point])

    return y_hat_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = Loading_Data()
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    #distance_df = Euclidean_Distance(X_train)

    # Save Data Frame
    #distance_df.to_csv("Distance_DF", sep=',', index=False, encoding='utf-8')

    # Load Distance Data Frame
    #distance_df = pd.read_csv("Distance_DF", sep=',')
    #
    #for i in range(10):
    #    test_digit = X_train[distance_df[i:i+1].index.values[0]: distance_df[i:i+1].index.values[0]+1]
    #    test_digit_image = test_digit.values.reshape((28, 28))
    #    plt.imshow(test_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
    #    plt.axis("off")
    #    plt.show()
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)

    print(y_hat_test)
    #counter = Counter(y_train[distance_df.index])
