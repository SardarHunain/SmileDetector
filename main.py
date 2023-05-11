import cv2

#face and smile clasifiers
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smileDetector = cv2.CascadeClassifier("haarcascade_smile.xml")

#grab webcam feed
webcam = cv2.VideoCapture(0)


while True:
#read current frame from webcam
    successful_frame_read, frame = webcam.read()
#if their an error abort
    if not successful_frame_read:
        break

#make gray
    frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#detect faces after making gray
    faces = face_detector.detectMultiScale(frame_grayscale,1.3,5)
#loop for running smile detection on all of the detected faces
    for(x,w,w,h) in faces:
        #draw rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)

        #create face sub image
        face = frame[y:y+h,x:x+w]

        #grayscale the face
        face_grayscale = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        #detect smiles in the face
        smile = smileDetector.detectMultiScale(face_grayscale,1.7,20)

        #label this face as smiling if smile found
        if(len(smile)>0):
            cv2.putText(frame,'smiling',(x,y+h+40),fontScale = 3,fontFace= cv2.FONT_HERSHEY_PLAIN,color = (255,255,255))

    #Show the current frame
    cv2.imShow('why so serious?',frame)
    #stop if Q is pressed
    key = cv2.waitkey(1)
    if(key==81 or key==113):
        break

#Cleanup
webcam.release()
cv2.destroyAllWindows()

