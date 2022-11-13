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

pathSave = 'dataset3/'


num_data = 1

while True:
        ret, frame = cap.read()
        frame_scaled = np.copy(frame)
        # frame_scaled = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        frame_save = np.copy(frame_scaled)

        gray = cv.cvtColor(frame_scaled, cv.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        eyes = eye_detector.detectMultiScale(gray, 1.3, 3)

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

                                # cari mata kiri atau kanan
                                if eyes[0][0] < eyes[1][0]:
                                        eye_left = eyes[0]
                                        eye_right = eyes[1]
                                else:
                                        eye_left = eyes[1]
                                        eye_right = eyes[0]

                                jarak = input("Masukkan jarak (cm): ")
                                

                                

                                with open(pathSave+str(num_data)+'_'+ jarak + '.txt','w') as f:
                                        f.write(str(faces[0][0])+','+str(faces[0][1])+','+str(faces[0][2])+','+str(faces[0][3])+',')
                                        
                                        if(len(eyes) == 2):
                                                f.write(str(eye_left[0])+','+str(eye_left[1])+','+str(eye_left[2])+','+str(eye_left[3])+',')
                                                f.write(str(eye_right[0])+','+str(eye_right[1])+','+str(eye_right[2])+','+str(eye_right[3])+',')
                                        
                                        f.write(jarak+'\n')        

                                cv.imwrite(pathSave+str(num_data)+'_'+ jarak + '.jpg', frame_save)



                                print("---- Gambar tersimpan ----")
                                print("faces: ", faces)
                                print("eyes left: ", eye_left)
                                print("eyes right: ", eye_right)

                                num_data += 1
                       

                                print("----- Kembali ke menu utama -----")
                                break
                        if cv.waitKey(1) & 0xFF == ord('n'):
                                print("---- Gambar tidak tersimpan ----")
                                print("----- Kembali ke menu utama -----")
                                break

                        

                



cap.release()
cv.destroyAllWindows()


