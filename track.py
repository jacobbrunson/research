import numpy as np
import cv2

def compare_pts(set1, img1, set2, img2):
	if len(set1) > len(set2):
		a = set1
		b = set2
	else:
		a = set2
		b = set1

	total_similar = 0

	for p1 in a:
		p1 = p1[0]
		roi1 = img1[p1[0]-4:p1[0]+4, p1[1]-4:p1[1]+4]
		
		if cv2.mean(roi1) == (0.0, 0.0, 0.0, 0.0):
			pass
		for p2 in b:
			p2 = p2[0]
			roi2 = img2[p2[0]-4:p2[0]+4, p2[1]-4:p2[1]+4]

	return total_similar / len(a)



cap = cv2.VideoCapture('/Users/jacobbrunson/Research/newmov.mov')
cap.read()
color = np.random.randint(0, 255, (100, 3))

contourCount = -1
p0 = None
mask = None
old_frame = None
old_gray = None

unique = []

while True:
	ret, next_frame = cap.read()

	if not ret:
		break

	next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

	# Pre-processing for contour detection
	hsv = cv2.cvtColor(next_frame, cv2.COLOR_BGR2HSV)
	wat = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([255, 80, 255]))
	m = cv2.bitwise_not(wat);
	m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
	wat = cv2.bitwise_and(next_frame, next_frame, mask=m);

	# Find the contours
	contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [x for x in contours if cv2.contourArea(x) > 1000]

	#An object entered or exited the scene, so we need to find new key points
	if len(contours) != contourCount:
		contourCount = len(contours)
		#print contourCount
		p0 = None
		old_frame = next_frame
		old_gray = next_gray
		mask = np.zeros_like(old_frame)

		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			roi = np.zeros(old_gray.shape, np.uint8)
			roi[y:(y+h), x:(x+w)] = 255
			
			pic = old_frame[y:(y+h), x:(x+w)]
			pic_hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
			pic_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
			sift = cv2.SIFT()
			kp1, desc1 = sift.detectAndCompute(pic_gray, None)
			hist1 = cv2.normalize(cv2.calcHist([pic_hsv], [0, 1], None, [180, 255], [0, 180, 0, 255]))
			tmp = cv2.goodFeaturesToTrack(pic_gray, mask=None, maxCorners=5, qualityLevel=0.5, minDistance=7, blockSize=7)
			
			bf = cv2.BFMatcher()

			best_matches = 0
			best_matches_i = 0
			best_hist = 0
			best_hist_i = 0
			best_pts = 0
			best_pts_i = 0
			for i in range(len(unique)):
				pts = unique[i]["pts"]
				#asdf = compare_pts(tmp, pic, pts, unique[i]["pics"][len(unique[i]["pics"])-1])
				matches = bf.knnMatch(desc1, unique[i]["desc"], k=2)
				d = cv2.compareHist(hist1, unique[i]["hist"], cv2.cv.CV_COMP_CORREL)
				#if asdf > best_pts:
					#best_pts = asdf
					#best_pts_i = i
				if d > best_hist:
					best_hist = d
					best_hist_i = i
				good = []
				if len(matches) > 0 and len(matches[0]) > 1:
					for m, n in matches:
						if m.distance < n.distance*0.75:
							good.append([m])
				if len(good) > best_matches:
					best_matches = len(good)
					best_matches_i = i


			if not desc1 is None:
				if float(best_matches)/float(len(kp1)) < 0.1 or best_hist < 0.3:
					unique.append({"desc": desc1, "hist": hist1, "pts": tmp, "pics":[pic]})
					print "Total unique objects" + str(len(unique))
				else:
					if best_matches / 10 > best_hist:
						s = best_hist_i
					else:
						s = best_hist_i
					unique[s]["desc"] = desc1
					unique[s]["hist"] = hist1
					unique[s]["pts"] = tmp
					unique[s]["pics"].append(pic)


			
			# Convert from ROI space to image space
			if not tmp is None:
				for j in range(len(tmp)):
					tmp[j][0][0] += x
					tmp[j][0][1] += y

			if p0 is None:
				p0 = tmp
			elif not tmp is None:
				p0 = np.concatenate((p0, tmp))
		ret, frame = cap.read()
	else:
		frame = next_frame

	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	# draw the tracks
	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
		cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

	img = cv2.add(frame, mask)

	cv2.imshow('frame', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
print "FINAL COUNT: %d" % len(unique)
for i in range(len(unique)):
	for j in range(len(unique[i]["pics"])):
		cv2.imshow("%d - %d" % (i, j), unique[i]["pics"][j])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cap.release()
		
