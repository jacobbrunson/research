import math

import numpy as np
import cv2

def find_contours(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	wat = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([255, 80, 255]))
	m = cv2.bitwise_not(wat);
	m = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=3)
	m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=3)
	#cv2.imshow("mask", m)
	#cv2.waitKey(0)
	contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return [x for x in contours if cv2.contourArea(x) > 2000]

def make_hist(img, contour):

	def compute_product(p, a, b):
		if a[0] == b[0]:
			k = 0
		else:
			k = (a[1] - b[1]) / (a[0] - b[0])

		j = a[1] - k * a[0]
		return k * p[0] - p[1] + j

	def in_rect(p, vertices):
		pro = []

		for i in range(4):
			pro.append(compute_product(p, vertices[i], vertices[(i+1) % 4]))

		return pro[0]*pro[2] < 0 and pro[1]*pro[3] < 0

	x, y, w, h = cv2.boundingRect(contour)

	return cv2.calcHist(img[y:y+h, x:x+w], [0, 1], None, [180, 255], [0, 180, 0, 255])

	vertices = cv2.cv.BoxPoints(cv2.minAreaRect(contour))
	pixels = []

	for q in range(y, y+h):
		for r in range(x, x+w):
			p = (r, q)
			pixel = img[q][r]
			if in_rect(p, vertices):
				pixels.append(pixel)

	return cv2.normalize(cv2.calcHist(np.array(pixels), [0, 1], None, [180, 255], [0, 180, 0, 255]))

class Object:

	def __init__(self, contour, position, hist, pic):
		self.contour = contour
		self.position = position
		self.hist_total = hist
		self.pics = [pic]
		self.is_dupe = False

	def update(self, contour, position, hist, pic):
		self.contour = contour
		self.position = position
		self.hist_total += hist
		self.pics.append(pic)

	def get_hist(self):
		return self.hist_total / len(self.pics)

cap = cv2.VideoCapture('/Users/jacobbrunson/Research/newnewmov.mov')
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
current_frame = 0
ret, old = cap.read()
old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
old_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV)

contours = find_contours(old)
unique_contours = []
objects = []
p0 = None

for contour in contours:
	x, y, w, h = cv2.boundingRect(contour)
	cx = x + w/2
	cy = y + h/2
	roi = old[y:(y+h), x:(x+w)]
	unique_contours.append((cx, cy))
	objects.append(Object(contour, (cx, cy), make_hist(old_hsv, contour), roi))

overlay = np.zeros_like(old)

old_contours = list(unique_contours)
old_objects = list(objects)

while True:
	ret, frame = cap.read()
	current_frame += 1

	if cv2.waitKey(1) & 0xFF == 27 or not ret:
		break

	print "Processing frame %d of %d [%d%%]" % (current_frame, length, int(float(current_frame)/length*100))

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	img = frame.copy()

	contours = find_contours(frame)

	old_found = []
	new_contours = []

	for i in range(len(contours)):
		contour = contours[i]
		x, y, w, h = cv2.boundingRect(contour)
		cx = x + w/2
		cy = y + h/2
		best = (None, 99999)
		for j in range(len(old_objects)):
			if j in old_found:
				continue
			u = old_objects[j]
			dist = math.sqrt((cx-u.position[0])**2 + (cy-u.position[1])**2)
			if dist < best[1]:
				best = (u, dist)
		if best[1] < 40:
			old_found.append(best[0])
			best[0].update(contour, (cx, cy), make_hist(hsv, contour), frame[y:(y+h), x:(x+w)])
		else:
			new_contours.append(i)


	old_notfound = []
	for o in old_objects:
		if not o in old_found:
			old_notfound.append(o)


	new_objects = list(old_found)

	for i in new_contours:
		contour = contours[i]

		x, y, w, h = cv2.boundingRect(contour)
		cx = x + w/2
		cy = y + h/2
		roi = frame[y:(y+h), x:(x+w)]
		hist = make_hist(hsv, contour)

		best = (None, 999999)
		for o in objects:
			if o in old_found:
				continue
			r = cv2.compareHist(hist, o.get_hist(), cv2.cv.CV_COMP_BHATTACHARYYA)
			print r
			if r < 0.9 and r < best[1]:
				best = (o, r)

		if not best[0] is None:
			best[0].update(contour, (cx, cy), hist, roi)
			new_objects.append(best[0])
			cv2.circle(img, best[0].position, 20, (255, 255, 0), 40)
		else:
			obj = Object(contour, (cx, cy), hist, roi)
			objects.append(obj)
			new_objects.append(obj)
			cv2.circle(img, obj.position, 20, (0, 255, 0), 40)

	for o in old_notfound:
		cv2.circle(img, o.position, 20, (0, 0, 255), 40)
		pass

	old_gray = gray.copy()

	old_contours = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		cx = x+w/2
		cy = y+h/2
		old_contours.append((cx, cy))

	old_objects = new_objects

	cv2.imshow("img", img)



for object in objects:
	dimensions = []
	color = []
	for pic in object.pics:
		dimensions.append(pic.shape)
		mean = cv2.mean(cv2.cvtColor(pic, cv2.COLOR_BGR2HSV))
		color.append(mean)
	dimensions = np.array(dimensions)
	color = np.array(color)

	object.shape_mean = dimensions.mean(axis=0)
	object.shape_std_dev = dimensions.std(axis=0)

	object.color_mean = color.mean(axis=0)
	object.color_std_dev = color.std(axis=0)


def determine(object, pic):
	color = cv2.mean(cv2.cvtColor(pic, cv2.COLOR_BGR2HSV))
	shape = pic.shape
	return np.all(np.absolute(shape - object.shape_mean) <= object.shape_std_dev*1) and np.all(np.absolute(color - object.color_mean) <= object.color_std_dev*1)

asdf = 0
hjkl = 0
for a in objects:
	asdf += 1
	hjkl += len(a.pics)
	for b in objects:
		if a is b or a.is_dupe or b.is_dupe:
			continue
		if np.all(np.absolute(a.shape_mean - b.shape_mean) < 20) and np.all(np.absolute(a.color_mean - b.color_mean) < 20):
			b.is_dupe = True
			a.pics += b.pics
			print "DUPE!"

print len(objects)
objects = [x for x in objects if not x.is_dupe]
print len(objects)

cv2.waitKey(0)
cv2.destroyAllWindows()
count = 0
total = 0
print hjkl/asdf/2
for object in objects:
	if len(object.pics) < hjkl/asdf/2:
		continue
	total += 1
	count += len(object.pics)
	for pic in object.pics:
		cv2.imshow("Object", pic)
	cv2.waitKey(0)

print "Total objects: %d" % total
print "Total iamges: %d" % count
print "Average # images per object: %.2f" % (float(count)/float(total))

cap.release()
cv2.destroyAllWindows()
