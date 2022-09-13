import cv2
import numpy as np
import face_recognition
image_kiar =  face_recognition.load_image_file('kiara_advani_train.jpg')
image_kiar = cv2.cvtColor(image_kiar,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('katrina.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
fac_loc = face_recognition.face_locations(image_kiar)[0]
encodkiara = face_recognition.face_encodings(image_kiar)[0]
cv2.rectangle(image_kiar,(fac_loc[3],fac_loc[0]),(fac_loc[1],fac_loc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgtest)[0]
encodtest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(fac_loc[3],fac_loc[2]),(fac_loc[0],fac_loc[0]),(255,0,255),2)
cv2.imshow('Kiara Advani',image_kiar)
cv2.imshow('Kiara Test',imgtest)
results = face_recognition.compare_faces([encodkiara],encodtest)
print(results)
cv2.waitKey(0)