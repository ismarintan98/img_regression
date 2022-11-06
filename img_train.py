# ------- Program untuk train data -------
# author: Moh Ismarintan Zazuli
# email: ismarintan16@mhs.ee.its.ac.id


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

img_mirror = cv.flip(img, 1)
faces_mirror = np.copy(faces)
eyes1_mirror = np.copy(eyes1)
eyes2_mirror = np.copy(eyes2)

faces_mirror[0] = img.shape[1] - faces[0] - faces[2]
eyes1_mirror[0] = img.shape[1] - eyes1[0] - eyes1[2]
eyes2_mirror[0] = img.shape[1] - eyes2[0] - eyes2[2]

cv.rectangle(img_mirror, (int(faces_mirror[0]), int(faces_mirror[1])), (int(faces_mirror[0]+faces_mirror[2]), int(faces_mirror[1]+faces_mirror[3])), (0, 255, 0), 2)
cv.rectangle(img_mirror, (int(eyes1_mirror[0]), int(eyes1_mirror[1])), (int(eyes1_mirror[0]+eyes1_mirror[2]), int(eyes1_mirror[1]+eyes1_mirror[3])), (0, 0, 255), 2)
cv.rectangle(img_mirror, (int(eyes2_mirror[0]), int(eyes2_mirror[1])), (int(eyes2_mirror[0]+eyes2_mirror[2]), int(eyes2_mirror[1]+eyes2_mirror[3])), (0, 0, 255), 2)


cv.imshow('img', img)
cv.imshow('img_mirror', img_mirror)

cv.waitKey(0)


