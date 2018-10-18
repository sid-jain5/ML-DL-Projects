# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:26:35 2018

@author: siddh
https://github.com/PukkaPad/Recognizing-handwritten-digits-KNN/blob/master/MNIST_KNN_python.py
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#Splitting Train/Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)

#Splitting train set into validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.1, random_state = 84)

print("training data points: {}".format(len(y_train)))
print("validation data points: {}".format(len(y_valid)))
print("testing data points: {}".format(len(y_test)))

from sklearn.neighbors import KNeighborsClassifier

kVals = range(1, 30, 2)
accuracies = []

for k in range(1, 30, 2):
    print("train the classifier with the current value of {}".format(k))
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    #evaluate the model and print the accuracies list
    score = model.score(X_valid, y_valid)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
    
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
    accuracies[i] * 100))

kVals[i]
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(X_train, y_train)

# Predict labels for the test set
predictions = model.predict(X_test)

# Evaluate performance of model for each of the digits
from sklearn.metrics import classification_report
print("EVALUATION ON TESTING DATA")
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)



sum = 0
for i in range (0,10):
    sum = sum + cm[i][i]
    
print(sum)

sum/len(y_test)

z = X_test[254]
z = z.reshape(28,28)
z.shape
plt.imshow(z, cmap = 'Greys')

"""from skimage import exposure
import imutils
import cv2
for i in np.random.randint(0, high=len(y_test), size=(5,)):
    # np.random.randint(low, high=None, size=None, dtype='l')
    image = X_test[i]
    prediction = model.predict(image)[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels for better visualization
    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)"""
