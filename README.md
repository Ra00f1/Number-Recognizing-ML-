# Number Recognizing(ML)
 
A prototype program to test the KNeighborsClassifier model on a live video feed.

Mnist data is used to train the model. The data is shifted by one pixel to add more data to the training set.

The program uses the cv2 library to capture the video feed from the webcam. The video feed is then processed to extract the region of interest (ROI) which is then passed to the model for prediction.

As this is a prototype I did not spend to much time on fine tuning and therefore it sometimes doesn't work well while using the webcam especially if the angle or the processed image  is not clear enough for the model.

###Sources:
1- https://towardsdatascience.com/how-to-build-knn-from-scratch-in-python-5e22b8920bd2
2- https://www.ibm.com/topics/knn#:~:text=Next%20steps-,K-Nearest%20Neighbors%20Algorithm,of%20an%20individual%20data%20point.
3- https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
4- https://stathwang.github.io/k-nearest-neighbors-from-scratch-in-python.html
5- https://www.geeksforgeeks.org/k-nearest-neighbours/

###Pandas:
7- https://pandas.pydata.org/docs/reference/api/pandas.concat.html

###OpenCV:
8- https://stackoverflow.com/questions/17987598/how-can-i-use-imshow-to-display-multiple-images-in-multiple-windows
6- https://stackoverflow.com/questions/64773918/creating-and-capturing-sub-region-on-web-cam-using-opencv
