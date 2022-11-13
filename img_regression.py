'''
 # @ Author: Moh Ismarintan Zazuli
 # @ Create Time: 2022-11-06 17:48:18
 # @ Description:
 '''


import cv2 as cv
import numpy as np



def main():
    #open camera
    cap = cv.VideoCapture(1)
    
    #display capture camera
    while True:
        ret, frame = cap.read()
        

        #resize frame to 50%
        frame2 = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
        
        
        

        
        cv.imshow('frame2', frame2)

        # print(np.shape(frame), np.shape(frame2_gray))


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # if pres 's' key, save image
        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.imwrite('dataset/test.jpg', frame2)
            print('save image')
    
    #release camera
    cap.release()



if __name__ == '__main__':
    main()



    

