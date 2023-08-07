import time
import  pandas as pd
import numpy as np
import multiprocessing

def calculate_distance(args):
    array, chunk = args
    print(chunk)
    dist = []
    start = time.time()
    for i in chunk:
        if i%100 == 0:
            end = time.time()
            print(i,"time: ", end - start)
            start = time.time()
        dist = [np.sqrt(np.sum((array[i] - array) ** 2, axis=1))]
    df = pd.DataFrame(dist, columns=['index', 'distance'])
    print(df)
    return df

def Euclidean_Distance(data):
    #data = data.head(100)
    array = data.to_numpy()
    length = array.shape[0]

    # Initialize the lists to store distances and indices
    train_distance_list = []
    train_id_counter = []
    num_processes = 10

    # Split the array indices into chunks for multiprocessing
    chunks = np.array_split(np.arange(length), num_processes)
    arguments = [(array, chunk) for chunk in chunks]

    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(calculate_distance, arguments)

    for chunk, distances in zip(chunks, results):
        print(chunk, "/", length)

        # Append the distances and corresponding indices to the lists
        train_distance_list.extend(distances)
        train_id_counter.extend([i for i in chunk] * length)

    d = {'index': train_id_counter, 'distance': train_distance_list}
    df = pd.DataFrame(d, columns=['index', 'distance'])
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

