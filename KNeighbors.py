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
import cv2

# Downloading MINST data
def Download_MINST():
    mnist = fetch_openml('mnist_784', parser="auto")
    print("Data Keys: ",mnist.keys())
    print("Data Download Completed!")
    return mnist

# Dividing data into training and testing
def Divide_Data(mnist):
    print("Dividing Data!")
    x, y = mnist["data"], mnist["target"]
    #print(x.shape)
    #print(y.shape)

    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

    return X_train, X_test, y_train, y_test

# Shifting images by one pixel to add more data
def Shift_Images(X_train, y_train):
    print("Shifting Images!")
    begin = time.time()
    trainX_aug = []
    trainY_aug = []

    for image, label in zip(np.array(X_train), y_train.values):
        # shift down
        shift_image = shift(image.reshape((28, 28)), [1, 0], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.extend(label)
        # shift up
        shift_image = shift(image.reshape((28, 28)), [-1, 0], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.extend(label)

        # shift right
        shift_image = shift(image.reshape((28, 28)), [0, 1], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.extend(label)
        # shift left
        shift_image = shift(image.reshape((28, 28)), [0, -1], cval=0)
        shift_image = shift_image.reshape([-1])
        trainX_aug.append(shift_image.round())
        trainY_aug.extend(label)
    print("Shifting Completed!")
    end = time.time()
    print("Time to run:", end - begin)
    begin = time.time()
    print("array to dataframe")

    # Only neede if you have 16GB RAM
    trainX_aug1 = pd.DataFrame(trainX_aug[:60000])
    trainX_aug2 = pd.DataFrame(trainX_aug[60000:120000])
    trainX_aug3 = pd.DataFrame(trainX_aug[120000:180000])
    trainX_aug4 = pd.DataFrame(trainX_aug[180000:240000])

    # The normal code
    #trainX_aug = pd.DataFrame(trainX_aug)

    trainX_aug = pd.concat([trainX_aug1, trainX_aug2, trainX_aug3, trainX_aug4], ignore_index=True)
    trainY_aug = pd.DataFrame(trainY_aug)

    trainY_aug.columns = ['class']
    x_cols = []

    # Correcting a mistake in the column names
    for i in trainX_aug.columns:
        x_cols.append("pixel"+str(i+1))
    trainX_aug.columns = x_cols

    # Connecting the old and the augmented data
    trainX_aug = pd.concat([trainX_aug, X_train], ignore_index=True)
    trainY_aug = pd.concat([trainY_aug, y_train], ignore_index=True)
    end = time.time()
    print("Time to run:", end - begin)
    return trainX_aug, trainY_aug

# Saving data to csv files(4 data files)
def Saving_Data(X_train, X_test, y_train, y_test):
    print("Saving Data!")
    X_train.to_csv("X_train", sep=',', index=False, encoding='utf-8')
    X_test.to_csv("X_test", sep=',', index=False, encoding='utf-8')
    y_train.to_csv("y_train", sep=',', index=False, encoding='utf-8')
    y_test.to_csv("y_test", sep=',', index=False, encoding='utf-8')
    print("Saving Completed!")

# Saving data to csv files(single data file)
def Save_Csv(file, filename):
    filename += '.csv'
    filename = "Data/" + filename
    file.to_csv(filename, sep=',', index=False, encoding='utf-8')
    print("Saved File: ", filename)

# Loading data from csv files(single data file)
def Load_Csv(filename):
    filename = "Data/" + filename + '.csv'
    print("Loading File: ", filename)
    return pd.read_csv(filename, sep=',')

# Loading data from csv files(4 data files)
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

# Testing model
def Test_Model(knn, X_test, y_test):
    print("Testing Model!")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Saving KNN model
def Save_Model(knn, filename):
    filename = "Trained/" + filename + '.pkl'
    joblib.dump(knn, filename)
    print("Saved Model: ", filename)

# Loading a saved KNN model
def Load_Model(filename):
    filename = "Trained/" + filename + '.pkl'
    print("Loading Model: ", filename)
    return joblib.load(filename)

def sketch_transform(image):  # in here you can do image filteration
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 150, 200]))
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations=1)
    thr = 255 - cv2.bitwise_and(dlt, msk)
    invert = cv2.bitwise_not(thr)
    return invert

def Webcam(knn):
    videoCaptureObject = cv2.VideoCapture(0)
    upper_left = (250, 250)
    bottom_right = (350, 350)

    result = True
    while (result):
        ret, image_frame = videoCaptureObject.read()

        # Rectangle marker
        r = cv2.rectangle(image_frame, upper_left, bottom_right, (200, 50, 300), 5)
        rect_img = image_frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

        sketcher_rect = rect_img
        sketcher_rect = sketch_transform(sketcher_rect)

        # Conversion for 3 channels to put back on original image (streaming)
        sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)

        # Replacing the sketched image on Region of Interest
        image_frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]] = sketcher_rect_rgb

        cv2.imshow("test", image_frame)

        #    break  # esc to quit
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k == ord('s'):
            cv2.imwrite("input_image.jpg", sketcher_rect_rgb)
            img = cv2.imread("input_image.jpg")

            # you can do those function inside the sketch_transform def

            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_image, (28, 28)).reshape(1, 28, 28, 1)
            # ax[2,1].imshow(gray_img.reshape(28, 28) , cmap = "gray")
            cv2.imshow("image", gray_img.reshape(28, 28))
            Test_Case_Array = gray_img.reshape(28, 28)
            Test_Case_Array = Test_Case_Array.reshape(1, -1)
            Test_Case = pd.DataFrame(Test_Case_Array)
            x_cols = []

            for i in Test_Case.columns:
                x_cols.append("pixel" + str(i + 1))
            Test_Case.columns = x_cols
            y_pred = knn.predict(Test_Case)
            print("predicted alphabet  = ", y_pred)

            print("if correct types Y/y else write the correct answer to add to the dataset")
            key = cv2.waitKey(0)
            key_list = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

            if key == 27:
                print("Escape hit, closing...")
                break
            if key == ord('Y') or key == ord('y'):
                print("Thank you for using our service")
            elif key in key_list:

                X_train_aug = Load_Csv('X_train_aug_V2')
                y_train_aug = Load_Csv('y_train_aug_V2')
                y_Test_Case = [key_list.index(key)]
                y_Test_Case = pd.DataFrame(y_Test_Case)
                y_Test_Case.columns = ['class']
                trainX_aug = pd.concat([X_train_aug, Test_Case], ignore_index=True)
                trainY_aug = pd.concat([y_train_aug, y_Test_Case], ignore_index=True)

                Save_Csv(trainX_aug, 'X_train_aug_V2')
                Save_Csv(trainY_aug, 'y_train_aug_V2')

                knn = K_Neighbors_Classifire(trainX_aug, trainY_aug.values.ravel())

                Save_Model(knn, 'knn_Library_V3')
                knn = Load_Model('knn_Library_V3')

if __name__ == "__main__":

    #Downloading and saving mnist
    #
    #mnist = Download_MINST()
    #X_train, X_test, y_train, y_test = Divide_Data(mnist)
    #Saving_Data(X_train, X_test, y_train, y_test)

    #X_train, X_test, y_train, y_test = Loading_Data()

    #X_train_aug, y_train_aug = Shift_Images(X_train, y_train)
    #X_train_aug = Load_Csv('X_train_aug')
    #y_train_aug = Load_Csv('y_train_aug')

    #knn = K_Neighbors_Classifire(X_train_aug, y_train_aug.values.ravel())
    #Test_Model(knn, X_test, y_test.values.ravel())
    #Save_Model(knn, 'knn_Library_V2')
    knn = Load_Model('knn_Library_V3')
    Webcam(knn)

    #for i  in range(20):
    #    print(y_test[i:i+1].values)
    #    pred = knn.predict(X_test[i:i+1])
    #    pred_percent = knn.predict_proba(X_test[i:i+1])
    #    print(pred, pred_percent)
    #Test_Model(knn, X_test, y_test)
