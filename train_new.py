# ------- Program untuk train data -------
# author: Moh Ismarintan Zazuli
# email: ismarintan16@mhs.ee.its.ac.id


import cv2 as cv
import numpy as np
import os
from nn_sin import nn_layers
import time

datasetPath = 'dataset/'

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
    
    layer1 = nn_layers(3,12,'sigmoid')
    layer2 = nn_layers(12,1,'purelin')
    learning_rate =  0.02

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



    for i in range(10000):
        print("Epoch ke-", i+1)
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
        
        print("x1", list_luasWajah[i], "x2", list_luasMataKiri[i], "x3", list_luasMataKanan[i], "y", list_trueJarak[i], "prediksi", prediksi)




        

    print("     ->  ---- Testing done ----")
    RMSE = np.sqrt(np.mean((listPrediksi - list_trueJarak)**2))
    print("     ->  ---- RMSE = ", RMSE, "----")


    

    

    
    
    




    

    