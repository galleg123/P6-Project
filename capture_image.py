import cv2
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


cam = cv2.VideoCapture(0)
"""
while True:
	ret, image = cam.read()
	#cv2.imshow('Imagetest',image)
	k = cv2.waitKey(1)
	if k != -1:
		break
"""
ret, image = cam.read()
cv2.imwrite(f'/home/pi/testimage_{dt_string.replace(" ","_").replace(":","-")}.jpg', image)
cam.release()
cv2.destroyAllWindows()
