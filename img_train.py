import cv2 as cv
import numpy as np


img = cv.imread('dataset/10.jpg')

txt_data = np.loadtxt('dataset/10.txt', delimiter=',')

faces = txt_data[0:4]
eyes1 = txt_data[4:8]
eyes2 = txt_data[8:12]

print(faces)
print(eyes1)
print(eyes2)

cv.rectangle(img, (int(faces[0]), int(faces[1])), (int(faces[0]+faces[2]), int(faces[1]+faces[3])), (0, 255, 0), 2) 
cv.rectangle(img, (int(eyes1[0]), int(eyes1[1])), (int(eyes1[0]+eyes1[2]), int(eyes1[1]+eyes1[3])), (0, 0, 255), 2) 
cv.rectangle(img, (int(eyes2[0]), int(eyes2[1])), (int(eyes2[0]+eyes2[2]), int(eyes2[1]+eyes2[3])), (0, 0, 255), 2)

cv.imshow('img', img)
cv.waitKey(0)


