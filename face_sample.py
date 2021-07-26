# In Face sample u can take the sample of the person face
"""
How to take the sample of face ?, follow the instruction
1)Make a folder of who u want to detect (in sample images)
2)In path_store_sampleimg give the similar name(name of folder ) b/t sample image and user
3)Take the sample images by running face_sample.py
4)In cascade_trained.py and face_recognize.py add your folder name in list People
5) train the cascade_trained() by running it
last) run your face_recognize.py

"""

# cv2 is the library of opencv
import cv2 as cv

# cascade classifier
cascade_face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# function to extract only face part


def face_extractor(img):
    # convert in gray scale (as grayscale simplifies the algorithm)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # face detection happen in rectangle shape
    faces_rect = cascade_face.detectMultiScale(gray, 1.3, 5)

    # if face not found return None
    if faces_rect is ():
        return None

    # if face found crop the face and return cropped face
    for (x, y, w, h) in faces_rect:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# turn on the pc video camera
cap = cv.VideoCapture(0)
# this is to count no of photo sample taken
count = 0

# in the while loop(multiple frame run)
while True:
    isTrue, frame = cap.read()
    # cv.imshow("frame", frame)

    # if face_extractor() return the croped face
    if face_extractor(frame) is not None:
        # increse the count
        count += 1
        # convert each frame into gray
        face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # store path + name + type of image
        path_store_sampleimg = 'sample_images/Brother/user' + str(
            count) + '.jpg'
        # store the click image in mentioned path
        cv.imwrite(path_store_sampleimg, face)
        
        # text which will count the number of photo clicked
        putText = cv.putText(face, str(count), (50, 100),
                             cv.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 2)
        
        cv.imshow("Click face Sample", face)
    else:
        print("Face not found")
    
    # with count ==100 it will automaticaly stop
    if cv.waitKey(2) == ord('q') or count == 100:
        break



# don't forget to do this
cap.release()
cv.destroyAllWindows()
