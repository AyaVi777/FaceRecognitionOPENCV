import cv2
import numpy as np
import face_recognition

imgLeclerc = face_recognition.load_image_file('ImagesBasic/Charles Leclerc.jpg')
imgLeclerc = cv2.cvtColor(imgLeclerc, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Charles Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgLeclerc)[0]
encodeLeclerc = face_recognition.face_encodings(imgLeclerc)[0]
cv2.rectangle(imgLeclerc,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,212),4)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,0,212),4)

results = face_recognition.compare_faces([encodeLeclerc],encodeTest)
faceDis = face_recognition.face_distance([encodeLeclerc],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,212),2)

cv2.imshow('Charles Leclerc' ,imgLeclerc)
cv2.imshow('Charles Test' ,imgTest)
cv2.waitKey(0)