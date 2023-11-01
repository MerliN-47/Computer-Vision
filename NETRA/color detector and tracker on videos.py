from collections import deque
import numpy as np
import argparse
import imutils
import cv2 as cv
import time
import pandas as pd
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())
greenLower = (25, 100, 100)
greenUpper = (70, 255, 255)
pts = deque(maxlen=args["buffer"])
camera = cv.VideoCapture(0)
Data_Features = ['x', 'y', 'time']
Data_Points = pd.DataFrame(data = None, columns = Data_Features , dtype = float)
start = time.time()

while True:
	(grabbed, frame) = camera.read()
	current_time = time.time() - start
	frame = imutils.resize(frame, width=1800)
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	mask = cv.inRange(hsv, greenLower, greenUpper)
	mask = cv.erode(mask, None, iterations=5)
	mask = cv.dilate(mask, None, iterations=5)
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	if len(cnts) > 0:
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if (radius < 300) & (radius > 10) : 
			cv.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv.circle(frame, center, 5, (0, 0, 255), -1)
			Data_Points.loc[Data_Points.size/3] = [x , y, current_time]
	pts.appendleft(center)
	for i in range(1, len(pts)):
		if pts[i - 1] is None or pts[i] is None:
			continue
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	cv.imshow("Frame", frame) ; cv.imshow("Mask", mask)
	key = cv.waitKey(1) & 0xFF
	if key == ord("q"):
		break
    
h = 0.2
X0 = -3
Y0 = 20
time0 = 0
theta0 = 0.3
Data_Points['x'] = Data_Points['x']- X0
Data_Points['y'] = Data_Points['y'] - Y0
Data_Points['time'] = Data_Points['time'] - time0
Data_Points['theta'] = 2 * np.arctan(Data_Points['y']*0.0000762/h)
Data_Points['theta'] = Data_Points['theta'] - theta0
plt.plot(Data_Points['theta'], Data_Points['time'])
plt.xlabel('Theta')
plt.ylabel('Time')
#Data_Points.to_csv('Data_Set.csv', sep=",")
#plt.savefig('Time_vs_Theta_Graph.svg', transparent= True)
camera.release()
cv.destroyAllWindows()