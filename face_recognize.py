# cv2 is the library of cv2(vision detection)
import cv2 as cv
# for mathmatical calculation(in array and matrix)
import numpy as npy

haar_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# list of people storing the name of(whoes photo face_recognize.py can detect)
people =['prianka copra','shown mendez','Justin bieber']

# read the face_trainded file
face_recognize = cv.face.LBPHFaceRecognizer_create()
face_recognize.read('face_trained.yml')

# turn on the default camera
cap = cv.VideoCapture(0)

# run the loop
while True : 
    isTrue, frame = cap.read()
    # cv.imshow("frame",frame)
    
    # convert each frome into gray scale
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # detect the face
    face_rect = haar_cascade.detectMultiScale(gray,1.2,5)
     
    
    for(x,y,w,h) in face_rect:
        # cut the detect region(crop the face)
        face_region = gray[y:y+h,x:x+w]
        
        # predict the crop face by .predict()
        label ,confidence = face_recognize.predict(face_region)
        # text the predicted person name
        cv.putText(frame, str(people[label]), (20,100), cv.FONT_HERSHEY_COMPLEX, 2.0, (0,0,255), thickness=2)
        # make a rectangle face boundry around the detected face
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)

     
    cv.imshow("recognize_face",frame)
    # press q to quit the program
    if cv.waitKey(2)==ord('q'):
        break

# don't forget to do this
cap.release()
cv.destroyAllWindows()
       