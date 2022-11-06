# ------- Program untuk train data -------
# author: Moh Ismarintan Zazuli
# email: ismarintan16@mhs.ee.its.ac.id


import cv2 as cv
import numpy as np


img = cv.imread('dataset/10.jpg')
img_mirror = cv.flip(img, 1)

txt_data = np.loadtxt('dataset/10.txt', delimiter=',')

faces = txt_data[0:4]
eyes_left = txt_data[4:8]
eyes_right = txt_data[8:12]


print(faces)
print(eyes_left)
print(eyes_right)

cv.rectangle(img, (int(faces[0]), int(faces[1])), (int(faces[0]+faces[2]), int(faces[1]+faces[3])), (0, 255, 0), 2) 
cv.rectangle(img, (int(eyes_left[0]), int(eyes_left[1])), (int(eyes_left[0]+eyes_left[2]), int(eyes_left[1]+eyes_left[3])), (255, 0, 0), 2)
cv.rectangle(img, (int(eyes_right[0]), int(eyes_right[1])), (int(eyes_right[0]+eyes_right[2]), int(eyes_right[1]+eyes_right[3])), (0, 0, 255), 2)


faces_mirror = np.copy(faces)
eyes_left_mirror = np.copy(eyes_right)
eyes_right_mirror = np.copy(eyes_left)

faces_mirror[0] = img.shape[1] - faces[0] - faces[2]
eyes_left_mirror[0] = img.shape[1] - eyes_right[0] - eyes_right[2]
eyes_right_mirror[0] = img.shape[1] - eyes_left[0] - eyes_left[2]

cv.rectangle(img_mirror, (int(faces_mirror[0]), int(faces_mirror[1])), (int(faces_mirror[0]+faces_mirror[2]), int(faces_mirror[1]+faces_mirror[3])), (0, 255, 0), 2)
cv.rectangle(img_mirror, (int(eyes_left_mirror[0]), int(eyes_left_mirror[1])), (int(eyes_left_mirror[0]+eyes_left_mirror[2]), int(eyes_left_mirror[1]+eyes_left_mirror[3])), (255, 0, 0), 2)
cv.rectangle(img_mirror, (int(eyes_right_mirror[0]), int(eyes_right_mirror[1])), (int(eyes_right_mirror[0]+eyes_right_mirror[2]), int(eyes_right_mirror[1]+eyes_right_mirror[3])), (0, 0, 255), 2)



cv.imshow('img', img)
cv.imshow('img_mirror', img_mirror)

cv.waitKey(0)


