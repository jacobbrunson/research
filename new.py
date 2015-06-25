import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/jacobbrunson/Research/newmov.mov')

ret, frame = cap.read()

print ret

cv2.waitKey(0)
cv2.destroyAllWindows()