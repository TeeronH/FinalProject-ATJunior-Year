import cv2

img_file = "img1.jpg"

img = cv2.imread(img_file)

player_classifier = 'players.xml'

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

player_tracker = cv2.CascadeClassifier(player_classifier)

player = player_tracker.detectMultiScale(gray_img)

print(player)

for (x,y,w,h) in player:
	cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 2)
	cv2.putText(img, 'Player', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow('my detection', img)

cv2.waitKey()



