import numpy as np
import cv2 

cam = cv2.VideoCapture(0)
HAAR_DIR = cv2.data.haarcascades
boxColor = (0, 196, 145)

faceCascades = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_alt2.xml")

while (True):
    ret, frame = cam.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascades.detectMultiScale(grayFrame, 
                                            scaleFactor=1.2, 
                                            minNeighbors=5,
                                            minSize=(30,30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w ,h) in faces:
        pocibleFaceGray = grayFrame[y:y+h, x:x+w]
        ###
        #  esta porcion de codigo puede servir para crear 
        #  el set de entrenamiento para hacer la identiciacion
        #  del rostro
        #  ---- INICIO ----
        ###

        cv2.imwrite("gray.png", pocibleFaceGray)
        cv2.imwrite("color.png", frame[y:y+h, x:x+w])

        ###        
        # ---- FIN ----
        # hasta aqui
        ###

        cv2.rectangle(frame, (x, y), (x + w, y + h), boxColor, 2)
    
    cv2.imshow('CAMARA', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
