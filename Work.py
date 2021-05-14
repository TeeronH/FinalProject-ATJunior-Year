
import cv2
import imutils
from PIL import Image
import numpy as np
import os.path

homedir = os.path.expanduser("~")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread(homedir + '/FinalProject-ATJunior-Year/FACE.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
	img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	first_image = Image.open('/Users/teeron/FinalProject-ATJunior-Year/FACE.jpg')
	second_image = Image.open('/Users/teeron/FinalProject-ATJunior-Year/Shrek1.jpg')
	second_image.thumbnail((w, h))
	first_image.paste(second_image, (x,y))

   
first_image.show()

'''
cv2.imshow('Git Shreked',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	shrekImage = (point(x,y), "Shrek1.png")

cv2.imshow("Faces found", image)
cv2.waitKey(0)
'''
