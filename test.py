import cv2
import numpy as np
import joblib

videoCaptureObject = cv2.VideoCapture(0)

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

upper_left = (200, 200)
bottom_right = (350, 350)

result = True

while (result):
    knn = Load_Model("knn_Library_V2")
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
        gray_img = cv2.resize(gray_image, (28, 28)).reshape(1,28,28,1)
        #ax[2,1].imshow(gray_img.reshape(28, 28) , cmap = "gray")
        cv2.imshow("image", gray_img.reshape(28, 28))

        y_pred = knn.predict_classes(img)
        print("predicted alphabet  = ", y_pred)