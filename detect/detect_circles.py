# USAGE
# python detect_circles.py --image images/simple.png

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])
output_sv = image.copy()
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 75)
r_max = 0
x_max = 0
y_max = 0

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	print(circles)
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	#find max radius
	x,y,r = max(circles, key=lambda x: x[2])
	print(x,y,r)
	output_sv = output_sv[(y+r-2*r):(y+r), (x-r):(x-r+2*r)]
	# show the output image
	cv2.imwrite('test.png', output_sv)
	cv2.imshow("output", np.hstack([image, output]))
	cv2.imshow("cropped", output_sv)
	cv2.waitKey(10000)