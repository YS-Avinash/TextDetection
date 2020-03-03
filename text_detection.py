# importing packages
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

# load the input image and resizing the image by ignoring aspect ratio
image = cv2.imread(input("Enter filename:"))
orig = image.copy()
(H, W) = image.shape[:2]
(newW, newH) = (320,320)
rW = W / float(newW)
rH = H / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# defining the two output layer names
# the first is the output probabilities and the
# second is used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

net = cv2.dnn.readNet("text_detection.pb")

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

print("Text detection took {:.6f} seconds".format(end - start))
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []
#loop over each row
for y in range(0, numRows):
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		if scoresData[x] < 0.5:
			continue
		
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

boxes = non_max_suppression(np.array(rects), probs=confidences)

# looping the bounding boxes
for (startX, startY, endX, endY) in boxes:
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
