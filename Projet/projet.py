import cv2   
import numpy as np
import picamera
import io
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

    #capture webcam
    #cap=cv2.VideoCapture(0)

    # Initialisation de la caméra
cap=picamera.PiCamera()
cap.resolution=(640,480)
cap.framerate=32
rawCapture=PiRGBArray(cap,size=(640,480))

    #Acquisition d'une image
    #cap.capture(rawCapture,format='bgr')
#img=rawCapture.array
#cap.start_preview()
for frame in cap.capture_continuous(rawCapture,format="bgr",use_video_port=True):
#while(1):
        img=frame.array
        #_, img = cap.read()
        
        #convertie frame(img  BGR) en HSV (hue-saturation-value)
        #hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        #definir la gamme de couleur rouge
        rouge_lower=np.array([136,87,111],np.uint8)
        rouge_upper=np.array([180,255,255],np.uint8)

        #définir la gamme de couleur bleu
        bleu_lower=np.array([99,115,150],np.uint8)
        bleu_upper=np.array([110,255,255],np.uint8)
        
        #definie la gamme de couleur jaune
        jaune_lower=np.array([22,60,200],np.uint8)
        jaune_upper=np.array([60,255,255],np.uint8)

        #definie la gamme de couleur blanch
        blanc_lower=np.array([254,254,254],np.uint8)
        blanc_upper=np.array([255,255,255],np.uint8)

        #retrouver la gamme de couleurs rouge, bleue , jaune et blanc dans l'image
        rouge=cv2.inRange(hsv,rouge_lower,rouge_upper)
        bleu=cv2.inRange(hsv,bleu_lower,bleu_upper)
        jaune=cv2.inRange(hsv,jaune_lower,jaune_upper)
        blanc=cv2.inRange(hsv,blanc_lower,blanc_upper)

        
        #Transformation morphologique, Dilatation    
        kernal = np.ones((5 ,5), "uint8")

        rouge=cv2.dilate(rouge, kernal)
        res=cv2.bitwise_and(img, img, mask = rouge)

        bleu=cv2.dilate(bleu,kernal)
        res1=cv2.bitwise_and(img, img, mask = bleu)

        jaune=cv2.dilate(jaune,kernal)
        res2=cv2.bitwise_and(img, img, mask = jaune)   

        blanc=cv2.dilate(blanc,kernal)
        res3=cv2.bitwise_and(img, img, mask = blanc) 


        #Suivi de la couleur rouge
        (_,contours,hierarchy)=cv2.findContours(rouge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                
                x,y,w,h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(img,"Rouge",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
                
        #Suivi de la couleur bleu
        (_,contours,hierarchy)=cv2.findContours(bleu,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,"Bleu",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

        #Suivi de la couleur jaune
        (_,contours,hierarchy)=cv2.findContours(jaune,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img,"Jaune",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  

        #Suivi de la couleur blanche
        (_,contours,hierarchy)=cv2.findContours(blanc,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour) 
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
                cv2.putText(img,"blanc",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))  




        cv2.imshow("Detection des Couleurs",img)
        rawCapture.truncate(0)
      
        if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.stop_preview()
                cap.release()
                cv2.destroyAllWindows()
                break  
              

        
