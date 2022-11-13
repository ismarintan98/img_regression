# ------- Program untuk train data -------
# author: Moh Ismarintan Zazuli
# email: ismarintan16@mhs.ee.its.ac.id


import cv2 as cv
import numpy as np
import os
from nn_sin import nn_layers
import time

datasetPath = 'dataset2/'
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






L_face = np.zeros(numData)
L_eye_left = np.zeros(numData)
L_eye_right = np.zeros(numData)

dat = []

for i in range(numData):
        print("---- Data ke-", i+1, "----")

        img = list_img[i]
        koordinat = list_koordinat[i]

        faces = koordinat[0:4]
        eye_left = koordinat[4:8]
        eye_right = koordinat[8:12]
        jarak = koordinat[12]

        x1 = int(faces[0])
        y1 = int(faces[1])
        x2 = int(faces[0] + faces[2])
        y2 = int(faces[1] + faces[3])

        #komponen deteksi [luas_wajah,luas_mata_kiri,luas_mata_kanan,jarak]
        komponen = [faces[2]*faces[3], eye_left[2]*eye_left[3], eye_right[2]*eye_right[3], jarak]

        print("------ Komponen ------")
        print("luas wajah:", komponen[0])
        print("luas mata kiri:", komponen[1])
        print("luas mata kanan:", komponen[2])
        print("jarak :", komponen[3])


        print("------ Koordinat ------")
        print("koordinat wajah: ", x1, y1, x2, y2)
        cv.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
        print("koordinat mata kiri: ", int(eye_left[0]), int(eye_left[1]), int(eye_left[0] + eye_left[2]), int(eye_left[1] + eye_left[3]))
        cv.rectangle(img, (int(eye_left[0]), int(eye_left[1])), (int(eye_left[0] + eye_left[2]), int(eye_left[1] + eye_left[3])), (0,255,0), 2)
        print("koordinat mata kanan: ", int(eye_right[0]), int(eye_right[1]), int(eye_right[0] + eye_right[2]), int(eye_right[1] + eye_right[3]))
        cv.rectangle(img, (int(eye_right[0]), int(eye_right[1])), (int(eye_right[0] + eye_right[2]), int(eye_right[1] + eye_right[3])), (0,0,255), 2)
        
        cv.imshow('img:'+ str(i+1), list_img[i])
        cv.waitKey(0)
        cv.destroyAllWindows()

        dat.append(komponen)


print("---- Data Training ----")
print(dat)
print(np.shape(dat))

luas_wajah = np.zeros(numData)
luas_mata_kiri = np.zeros(numData)
luas_mata_kanan = np.zeros(numData)
jarak = np.zeros(numData)


for i in range(numData):
        luas_wajah[i] = dat[i][0]
        luas_mata_kiri[i] = dat[i][1]
        luas_mata_kanan[i] = dat[i][2]
        jarak[i] = dat[i][3]


#normalisasi
luas_wajah_norm = luas_wajah/np.max(luas_wajah)
luas_mata_kiri_norm = luas_mata_kiri/np.max(luas_mata_kiri)
luas_mata_kanan_norm = luas_mata_kanan/np.max(luas_mata_kanan)
jarak_norm = jarak/np.max(jarak)


print("---- Data Training Test ----")
print(luas_wajah)
print(luas_mata_kiri)
print(luas_mata_kanan)
print(jarak)

input_layer = np.zeros((3, 1))
output_layer = np.zeros((1, 1))

layer1 = nn_layers(3,12,'sigmoid')
layer2 = nn_layers(12,1,'purelin')

learning_rate = 0.00055

predic_denorm = np.zeros(numData)

print("------ Data Training ------")
for i in range(numData):
        print("---- Data ke-", i+1, "----")
        print("luas wajah:", luas_wajah_norm[i])
        print("luas mata kiri:", luas_mata_kiri_norm[i])
        print("luas mata kanan:", luas_mata_kanan_norm[i])
        print("jarak :", jarak_norm[i])
        
        


time_start = time.time()

for i in range(1000):
        
        if i % 10000 == 0:
                print("---- Iterasi ke-", i+1, "----")

        for j in range(numData):
                input_layer[0] = luas_wajah_norm[j]
                input_layer[1] = luas_mata_kiri_norm[j]
                input_layer[2] = luas_mata_kanan_norm[j]
                output_layer[0] = jarak_norm[j]

                layer1.forward(input_layer)
                layer2.forward(layer1.output_layer)

                layer2.backwardEndLayer(output_layer, learning_rate)
                layer1.backwardMidLayer(layer2, learning_rate)

                # print("Ep:", i+1, "Data ke:", j+1, "Error:", layer2.error)

print("------ Data Testing ------")
for i in range(numData):
        input_layer[0] = luas_wajah_norm[i]
        input_layer[1] = luas_mata_kiri_norm[i]
        input_layer[2] = luas_mata_kanan_norm[i]
        output_layer[0] = jarak_norm[i]

        layer1.forward(input_layer)
        layer2.forward(layer1.output_layer)

        predic_denorm[i] = layer2.output_layer[0]*np.max(jarak)

        print("x1:", luas_wajah[i], "x2:", luas_mata_kiri[i], "x3:", luas_mata_kanan[i], "y:", jarak[i], "prediksi:", predic_denorm[i])


RMSE = np.sqrt(np.mean(np.square(layer2.error)))
print("RMSE:", RMSE)

time_end = time.time()

print("Waktu eksekusi:", time_end - time_start, "detik")

