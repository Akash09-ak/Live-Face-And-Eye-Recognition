import numpy as np
import cv2

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector1= cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)

while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detector.detectMultiScale(gray, 1.3, 5)
	eyes = detector1.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	for (x,y,w,h) in eyes:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

	cv2.imshow('frame',img)
	#cv2.imwrite('frame',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
    
cap.release()
cv2.destroyAllWindows()
