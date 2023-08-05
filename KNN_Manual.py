import  pandas as pd
import numpy as np

def Euclidean_Distance(data):
    train_distance_list = []
    train_id_counter = []

    for i in range(len(data)):
        #if i%1000 == 0:
        #    print(i, "/", len(data))
        for j in range(len(data)):
            #distance = np.sqrt(np.sum(data.iloc[i] - data.iloc[j])**2)
            x = data.iloc[i]
            y = data.iloc[j]
            distance = np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
            train_distance_list.append(distance)
            train_id_counter.append(j)

        # make a dictionary
        d = {'index': train_id_counter, 'distance': train_distance_list}
        # convert dictionary to dataframe
        df = pd.DataFrame(d, columns=['index', 'distance'])
        # sort in ascending order by euclidean index
        df_sorted = df.sort_values(by='index')
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

    Euclidean_Distance(X_train)

