# ------- Program untuk train data -------
# author: Moh Ismarintan Zazuli
# email: ismarintan16@mhs.ee.its.ac.id


import cv2 as cv
import numpy as np
import os
from nn_sin import nn_layers
import time

datasetPath = 'dataset2/'

layer1 = nn_layers(3,12,'sigmoid')
layer2 = nn_layers(12,1,'purelin')
learning_rate =  0.01

def LoadData(path):

    listData = os.listdir(path)
    numData = int(len(listData)/2)

    if numData == 0:
        print("     ->  ---- Tidak ada data ----")
        print("     ->  ---- Silahkan buat data terlebih dahulu ----")
        exit()

    else:
        print("     ->Ditemukan", numData, "data")

        idx_img = 0
        idx_txt = 1
        list_img = []
        list_koordinat = []

        for i in range(numData):
            img_buff = cv.imread(path + listData[idx_img])
            list_img.append(img_buff)

            txt_buff = np.loadtxt(
                path + listData[idx_txt], delimiter=',')
            list_koordinat.append(txt_buff)

            idx_img += 2
            idx_txt += 2

        return list_img, list_koordinat, numData



def normalize(x, x_min, x_max):
    return 2*((x-x_min)/(x_max-x_min))-1

def denormalize(x, x_min, x_max):
    return 0.5 * (x + 1) * (x_max - x_min) + x_min


def playNNtoData():

    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    cap = cv.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        frame_scled = np.copy(frame)
        # frame_scled = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame_gray = cv.cvtColor(frame_scled, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        eye = eye_cascade.detectMultiScale(frame_gray, 1.3, 10)

        

        for (x, y, w, h) in faces:
            cv.rectangle(frame_scled, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            for (ex, ey, ew, eh) in eye:

                cv.rectangle(frame_scled, (ex, ey),
                             (ex+ew, ey+eh), (0, 255, 0), 2)


        
        #if number of face detected is 1

        faces_size = 0
        eye_L_size = 0
        eye_R_size = 0

        allDetect = 0

        if len(faces) == 1:
            faces_size = faces[0][2]*faces[0][3]
            allDetect += 1
        else :
            allDetect = -1
        
        if len(eye) == 2:
            eye_L_size = eye[0][2]*eye[0][3]
            eye_R_size = eye[1][2]*eye[1][3]
            allDetect += 1
        else :
            allDetect = -1

        if allDetect == 2:
            print("F", faces_size, "L", eye_L_size, "R", eye_R_size)
        



    
        cv.imshow('frame', frame_scled)




        if cv.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':

    print("==========================================")
    print("=========== NN IMG Regression ============")
    print("==========================================")


    #------------- Load Data ----------------
    print("# loading data...")
    Dataset = LoadData(datasetPath)
    
    lenDataset = Dataset[2]
    list_img = Dataset[0]
    list_koordinat = Dataset[1]

    list_luasWajah = np.zeros((lenDataset))
    list_luasMataKiri = np.zeros((lenDataset))
    list_luasMataKanan = np.zeros((lenDataset))
    list_trueJarak = np.zeros((lenDataset))

    

    for i in range(lenDataset):
        list_luasWajah[i] = list_koordinat[i][2] * list_koordinat[i][3]
        list_luasMataKiri[i] = list_koordinat[i][6] * list_koordinat[i][7]
        list_luasMataKanan[i] = list_koordinat[i][10] * list_koordinat[i][11]
        list_trueJarak[i] = list_koordinat[i][12]

    
    print(list_luasWajah)
    print(list_luasMataKiri)
    print(list_luasMataKanan)
    print(list_trueJarak)

    print("     ->  ---- Data loaded ----")

    #------------- Train Data ----------------
    print("# training data...")

    input_layer = np.zeros((3, 1))
    output_layer = np.zeros((1, 1))

    dataMax = [0,0,0,0]
    dataMin = [0,0,0,0]

    dataMax[0] = np.max(list_luasWajah)
    dataMax[1] = np.max(list_luasMataKiri)
    dataMax[2] = np.max(list_luasMataKanan)
    dataMax[3] = np.max(list_trueJarak)

    dataMin[0] = np.min(list_luasWajah)
    dataMin[1] = np.min(list_luasMataKiri)
    dataMin[2] = np.min(list_luasMataKanan)
    dataMin[3] = np.min(list_trueJarak)

    listPrediksi = np.zeros((lenDataset))



    for i in range(50000):
        if(i%1000 == 0):
            print("     -> epoch ke-",i)
        
        for j in range(lenDataset):
    
            input_layer[0] = normalize(list_luasWajah[j], dataMin[0], dataMax[0])
            input_layer[1] = normalize(list_luasMataKiri[j], dataMin[1], dataMax[1])
            input_layer[2] = normalize(list_luasMataKanan[j], dataMin[2], dataMax[2])
            output_layer[0] = normalize(list_trueJarak[j], dataMin[3], dataMax[3])

            layer1.forward(input_layer)
            layer2.forward(layer1.output_layer)

            layer2.backwardEndLayer(output_layer, learning_rate)
            layer1.backwardMidLayer(layer2, learning_rate)    



    print("     ->  ---- Training done ----")  

    #------------- Test Data ----------------
    print("# testing data...")

    for i in range(lenDataset):
        input_layer[0] = normalize(list_luasWajah[i], dataMin[0], dataMax[0])
        input_layer[1] = normalize(list_luasMataKiri[i], dataMin[1], dataMax[1])
        input_layer[2] = normalize(list_luasMataKanan[i], dataMin[2], dataMax[2])
        output_layer[0] = normalize(list_trueJarak[i], dataMin[3], dataMax[3])

        layer1.forward(input_layer)
        layer2.forward(layer1.output_layer)

        prediksi = denormalize(layer2.output_layer[0], dataMin[3], dataMax[3])
        listPrediksi[i] = prediksi
        
        print("x1", list_luasWajah[i], "x2", list_luasMataKiri[i], "x3", list_luasMataKanan[i], "y", list_trueJarak[i], "prediksi", prediksi[0])




        

    print("     ->  ---- Testing done ----")
    RMSE = np.sqrt(np.mean((listPrediksi - list_trueJarak)**2)) #type: ignore
    print("     ->  ---- RMSE = ", RMSE, "----")

    #------------- Play To Camera ----------------
    # playNNtoData()
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    cap = cv.VideoCapture(1)

    while True:
        ret, frame = cap.read()

        # frame_scled = cv.resize(frame, (0, 0), fx=1, fy=1)
        frame_scled = np.copy(frame)
        frame_gray = cv.cvtColor(frame_scled, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        eye = eye_cascade.detectMultiScale(frame_gray, 1.3, 5)

        

        for (x, y, w, h) in faces:
            cv.rectangle(frame_scled, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            for (ex, ey, ew, eh) in eye:

                cv.rectangle(frame_scled, (ex, ey),
                             (ex+ew, ey+eh), (0, 255, 0), 2)


        
        #if number of face detected is 1

        faces_size = 0
        eye_L_size = 0
        eye_R_size = 0

        allDetect = 0

        if len(faces) == 1:
            faces_size = faces[0][2]*faces[0][3]
            allDetect += 1
        else :
            allDetect = -1
        
        if len(eye) == 2:
            eye_L_size = eye[0][2]*eye[0][3]
            eye_R_size = eye[1][2]*eye[1][3]
            allDetect += 1
        else :
            allDetect = -1

        if allDetect == 2:
            # print("F", faces_size, "L", eye_L_size, "R", eye_R_size)
            input_layer[0] = normalize(faces_size, dataMin[0], dataMax[0])
            input_layer[1] = normalize(eye_L_size, dataMin[1], dataMax[1])
            input_layer[2] = normalize(eye_R_size, dataMin[2], dataMax[2])

            layer1.forward(input_layer)
            layer2.forward(layer1.output_layer)

            prediksi = denormalize(layer2.output_layer[0], dataMin[3], dataMax[3])
            print("Jarak", prediksi[0])
            # cv.putText(frame_scled, "Jarak : " + str(prediksi[0]), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame_scled, "Jarak : " + str(int(prediksi[0])), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame_scled,"L Wajah : " + str(faces_size), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame_scled,"L Mata Kiri : " + str(eye_L_size), (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame_scled,"L Mata Kanan : " + str(eye_R_size), (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        


        
        cv.imshow('frame', frame_scled)




        if cv.waitKey(1) & 0xFF == ord('q'):
            break




# playNNtoData()
    

    
    
    




    

    