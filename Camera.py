import time
import pygame.camera
import cv2
import numpy as np
from PIL import Image

pygame.camera.init()
cams = pygame.camera.list_cameras() #Camera detected or not
print("Cameras found: ",cams)
cam = pygame.camera.Camera(cams[0])
cam.start()
time.sleep(1)
print("Taking image...")
img = cam.get_image()
pygame.image.save(img,"filename.jpg")

print("Image taken")
cam.stop()
print("Processing image...")

img = cv2.imread("filename.jpg")
# Cvt to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

msk = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 150, 200]))
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dlt = cv2.dilate(msk, krn, iterations=1)
thr = 255 - cv2.bitwise_and(dlt, msk)
invert = cv2.bitwise_not(thr)
print("Image processed")
cv2.imwrite("Processed Image.jpeg", invert)
cv2.imshow("thr", invert)
cv2.waitKey(0)
cv2.destroyAllWindows()

im = Image.open("Processed Image.jpeg")

# Setting the points for cropped image
left = 275
top = 195
right = 400
bottom = 320

# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))
# Shows the image in image viewer
#im1.show()
im1 = im1.save("test.jpg")
im = Image.open("test.jpg")
im.save("test-600.jpg", dpi=(28,28))
#im.show()
#quit()


