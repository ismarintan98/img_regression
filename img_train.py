# ------- Program untuk train data -------
# author: Moh Ismarintan Zazuli
# email: ismarintan16@mhs.ee.its.ac.id


import cv2 as cv
import numpy as np
import os

datasetPath = 'dataset/'
listData = os.listdir(datasetPath)
numData = int(len(listData)/2)

if numData == 0:
        print("---- Tidak ada data ----")
        print("---- Silahkan buat data terlebih dahulu ----")
        exit()        
else:
        
        print("Ditemukan", numData, "data")

        idx_img = 0
        idx_txt = 1
        list_img = []
        list_koordinat = []
        for i in range(numData):
            img_buff = cv.imread(datasetPath + listData[idx_img])
            list_img.append(img_buff)

            txt_buff = np.loadtxt(datasetPath + listData[idx_txt], delimiter=',')
            list_koordinat.append(txt_buff)

            idx_img += 2
            idx_txt += 2



for i in range(numData):
        print("---- Data ke-", i+1, "----")

        img = list_img[i]
        koordinat = list_koordinat[i]

        faces = koordinat[0:4]
        eye_left = koordinat[4:8]
        eye_right = koordinat[8:12]

        x1 = int(faces[0])
        y1 = int(faces[1])
        x2 = int(faces[0] + faces[2])
        y2 = int(faces[1] + faces[3])

        #komponen deteksi [luas_wajah,luas_mata_kiri,luas_mata_kanan]
        komponen = [faces[2]*faces[3], eye_left[2]*eye_left[3], eye_right[2]*eye_right[3]]

        print("luas wajah:", komponen[0])
        print("luas mata kiri:", komponen[1])
        print("luas mata kanan:", komponen[2])


        print("koordinat wajah: ", x1, y1, x2, y2)
        cv.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
        print("koordinat mata kiri: ", int(eye_left[0]), int(eye_left[1]), int(eye_left[0] + eye_left[2]), int(eye_left[1] + eye_left[3]))
        cv.rectangle(img, (int(eye_left[0]), int(eye_left[1])), (int(eye_left[0] + eye_left[2]), int(eye_left[1] + eye_left[3])), (0,255,0), 2)
        print("koordinat mata kanan: ", int(eye_right[0]), int(eye_right[1]), int(eye_right[0] + eye_right[2]), int(eye_right[1] + eye_right[3]))
        cv.rectangle(img, (int(eye_right[0]), int(eye_right[1])), (int(eye_right[0] + eye_right[2]), int(eye_right[1] + eye_right[3])), (0,0,255), 2)
        
        cv.imshow('img:'+ str(i+1), list_img[i])
        cv.waitKey(0)
        
        