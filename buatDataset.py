# ------- Program untuk membuat dataset -------
# author: Moh Ismarintan Zazuli
# email: ismarintan16@mhs.ee.its.ac.id

import cv2 as cv
import numpy as np


face_detector = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_detector = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')


cap = cv.VideoCapture(1)


print("-------- Data Set Creator --------")
print("tekan 'q' untuk keluar")
print("tekan 's' untuk menyimpan gambar")




while True:
        ret, frame = cap.read()
        frame_scaled = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        frame_save = np.copy(frame_scaled)

        gray = cv.cvtColor(frame_scaled, cv.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        eyes = eye_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
                cv.rectangle(frame_scaled, (x,y), (x+w, y+h), (255,0,0), 2)

        for (x,y,w,h) in eyes:
                cv.rectangle(frame_scaled, (x,y), (x+w, y+h), (0,255,0), 2)
                

        cv.imshow('frame', frame_scaled)

        fps = cap.get(cv.CAP_PROP_FPS)


        if cv.waitKey(1) & 0xFF == ord('q'):
                print("---- Keluar ----")
                break

        if cv.waitKey(1) & 0xFF == ord('s'):
                print("---- Konfirmasi simpan y/n ? ----")

                while True:

                        if len(faces) != 1:
                                print("---- Tidak ada wajah atau lebih dari satu wajah ----")
                                print("tekan 'b' untuk kembali")
                                while True:
                                        if cv.waitKey(1) & 0xFF == ord('b'):
                                                break
                                print("----- Kembali ke menu utama -----")
                                break
                                
                        if len(eyes) != 2:
                                print("---- Mata tidak sama dengan dua ----")
                                print("tekan 'b' untuk kembali")
                                while True:
                                        if cv.waitKey(1) & 0xFF == ord('b'):
                                                break
                                print("----- Kembali ke menu utama -----")
                                                
                                break

                        if cv.waitKey(1) & 0xFF == ord('y'):

                                jarak = input("Masukkan jarak (cm): ")

                                

                                with open('dataset/'+ jarak + '.txt','w') as f:
                                        f.write(str(faces[0][0])+','+str(faces[0][1])+','+str(faces[0][2])+','+str(faces[0][3])+',')
                                        
                                        if(len(eyes) == 2):
                                                f.write(str(eyes[0][0])+','+str(eyes[0][1])+','+str(eyes[0][2])+','+str(eyes[0][3])+',')
                                                f.write(str(eyes[1][0])+','+str(eyes[1][1])+','+str(eyes[1][2])+','+str(eyes[1][3])+'\n')

                                cv.imwrite('dataset/' + jarak + '.jpg', frame_save)



                                print("---- Gambar tersimpan ----")
                                print("faces: ", faces)
                                print("eyes 1: ", eyes[0])
                                print("eyes 2: ", eyes[1])
                       

                                print("----- Kembali ke menu utama -----")
                                break
                        if cv.waitKey(1) & 0xFF == ord('n'):
                                print("---- Gambar tidak tersimpan ----")
                                print("----- Kembali ke menu utama -----")
                                break

                        

                



cap.release()
cv.destroyAllWindows()


