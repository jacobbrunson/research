import cv2
import numpy as np

colors = [
	(0, 0, 255),
	(0, 255, 0),
	(255, 0, 0),
	(0, 255, 255),
	(255, 0, 255),
	(255, 255, 0),
	(128, 0, 128),
	(128, 128, 0),
	(255, 255, 255),
	(128, 128, 128),
	(0, 0, 0)
]

points = [[]] * 11

cap = cv2.VideoCapture("movie.mov")

prev = []

while True:
	ret, frame = cap.read();

	#img = cv2.imread("woot.jpg");
	img = frame

	hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	rip = cv2.inRange(hsl, np.array([0, 0, 100]), np.array([255, 80, 255]))
	wat = cv2.bitwise_not(rip);

	kernel = np.ones((5, 5), np.uint8)

	wat = cv2.erode(wat, kernel, iterations=0)
	wat = cv2.dilate(wat, kernel, iterations=1)
	res = cv2.bitwise_and(img, img, mask=wat);

	contours, hierarchy = cv2.findContours(wat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours = [x for x in contours if cv2.contourArea(x) > 1000]

	next = [None] * max(len(contours), len(prev))
	i = 0

	# for k in range(0, len(colors)):
	# 	for point in points[k]:
	# 		print str(k) + " len: " + str(len(points[k]))
	# 		cv2.circle(img, (point[0], point[1]), 4, colors[k])

	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		cx = x + w/2
		cy = y + h/2

		best_dist = 9999999
		best = -1
		if len(prev) > 0:
			for j in range(0, len(prev)):
				if prev[j] is None:
					continue
				dist = (prev[j][0] - cx)**2 + (prev[j][1] - cy)**2
				if (dist < best_dist):
					best_dist = dist
					best = j
		if best == -1:
			color = colors[i]
			next[i] = (cx, cy)
			#points[i].append((cx, cy))
		else:
			color = colors[best]
			next[best] = (cx, cy)
			prev[best] = None
			points[best].append((cx, cy))

		roi = img[y:(y+h), x:(x+w)]
		cv2.rectangle(img, (x, y), (x+w, y+h), color, 3);
		cv2.circle(img, (cx, cy), 10, (color))
		i += 1

	prev = next		

	print "Detected " + str(len(contours)) + " objects"

	cv2.imshow("img", img);
	if cv2.waitKey(1) & 0xFF == 27:
		break
cv2.destroyAllWindows();