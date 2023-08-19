import cv2
import numpy as np

def show_webcam(mirror=False):
    n = 0
# use the device /dev/video{n} in this case /dev/video0
# On windows use the first connected camera in the device tree
    cam = cv2.VideoCapture(n)
    while True:
# read what the camera the images which are comming from the camera
# ret_val idk atm
        ret_val, img = cam.read()
# if mirror is true do flip the image
        if mirror:
            img = cv2.flip(img, 1)
# show the image with the title my webcam in a window
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 150, 200]))
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dlt = cv2.dilate(msk, krn, iterations=1)
        thr = 255 - cv2.bitwise_and(dlt, msk)
        invert = cv2.bitwise_not(thr)
        cv2.imshow('my webcam1', img)
        cv2.imshow('my webcam', invert)
        if cv2.waitKey(1) == ord('s'):
            cv2.imwrite("filename.jpg", invert)
# if you press ESC break out of the while loop and destroy all windows == free resources <3
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()