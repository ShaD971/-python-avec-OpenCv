import cv2   
import numpy as np

#capture webcam
cap=cv2.VideoCapture(0)

while(1):
    _, img = cap.read()
        
    #convertie frame(img  BGR) en HSV (hue-saturation-value)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #definir la gamme de couleur rouge
    red_lower=np.array([136,87,111],np.uint8)
    red_upper=np.array([180,255,255],np.uint8)

    #définir la gamme de couleur bleu
    blue_lower=np.array([99,115,150],np.uint8)
    blue_upper=np.array([110,255,255],np.uint8)
    
    #definie la game de couleur jaune
    yellow_lower=np.array([22,60,200],np.uint8)
    yellow_upper=np.array([60,255,255],np.uint8)

    #rouver la gamme de couleurs rouge, bleue et jaune dans l'image
    rouge=cv2.inRange(hsv, red_lower, red_upper)
    bleu=cv2.inRange(hsv,blue_lower,blue_upper)
    jaune=cv2.inRange(hsv,yellow_lower,yellow_upper)
    
    #Transformation morphologique, Dilatation    
    kernal = np.ones((5 ,5), "uint8")

    rouge=cv2.dilate(rouge, kernal)
    res=cv2.bitwise_and(img, img, mask = rouge)

    bleu=cv2.dilate(bleu,kernal)
    res1=cv2.bitwise_and(img, img, mask = bleu)

    jaune=cv2.dilate(jaune,kernal)
    res2=cv2.bitwise_and(img, img, mask = jaune)    


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
            cv2.putText(img,"jaune",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  
            

        cv2.imshow("Color Tracking",img)
  
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break  
          

    
