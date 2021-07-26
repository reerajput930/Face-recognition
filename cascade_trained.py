# cv2 is the library of cv2(vision detection)
import cv2 as cv
# for mathmatical calculation(in array and matrix)
import numpy as npy
# using for path join
import os

# cascade classifier for face detection
haar_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# list of people storing the name of(whoes photo face_recognize.py can detect)
people=['prianka copra','shown mendez','Justin bieber']
# directry where the photo is stored
dir = r'./sample_images/'

# empty list of features and labels
features = []
labels = []

# this is to trained it for face recognization
def create_training():

    # each folder in sample images
    for person in people:
        # join directry with person
        path = os.path.join(dir,person)
        # label it by folder
        label = people.index(person)
        
        # each img in th folder
        for img in os.listdir(path):
            # got the image path
            img_path = os.path.join(path,img)
            # reading the image
            img_array = cv.imread(img_path)
            if img_path is None:
                continue
            # coverting it inro gray(gray scale simplifies the algorithem)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            # detect happens in rectangle
            face_rect = haar_cascade.detectMultiScale(gray,1.2,5)
            
            # croping the detected face
            for (x,y,w,h) in face_rect:
                face_region = gray[y:y+h,x:x+w]
                # storing the face_region in features list
                features.append(face_region)
                # storing each face_region index
                labels.append(label)

# run the function 
create_training()
print('Training done ---------------')

# convert features into array
features = npy.array(features,dtype=object)
labels = npy.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
# train the recognizer by adding features and labels
face_recognizer.train(features,labels)

# save it so that we can use it in other file
face_recognizer.save("face_trained.yml")
npy.save("features.npy",features)
npy.save("labels.npy",labels)

