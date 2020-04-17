import math
import numpy as np  
import cv2 as cv
from sklearn.cluster import KMeans

"""
Takes 2 lines that represented by 2 points(a 2D list)
Returns the junction points if it exits, False otherwise
"""
def get_junction(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
	   return False, False

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return int(x), int(y)

"""
	Adds line to the destination frame
	Returns cartesian coordinates of the 2 points that belong to that line
"""
def add_line(lines, i, destination):
	rho = lines[i][0][0]
	theta = lines[i][0][1]
	a = math.cos(theta)
	b = math.sin(theta)
	x0 = a * rho
	y0 = b * rho
	pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
	pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
	cv.line(destination, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
	return pt1, pt2
"""
Paints a rectangle of 4x4 pixel size on the frame with the center given
as x and y.
"""
def mark_point(x, y, frame, rbg=(0,255,0)):
		p1 = (int(x-2),int(y-2))
		p2 = (int(x+2),int(y+2))     
		cv.rectangle(frame, p1, p2, rbg, 3, cv.LINE_AA)

"""
Marks all points using mark_point() function
"""
def mark_all(points, frame, rbg=(0,255,0)):
	for p in points:
		mark_point(p[0], p[1], frame,rbg)

"""
Given a list of lines that are represent by 2 2D points
returns all junction points by using get_junction() for each
pair of points
"""
def junctions(lines):
	intersections = []
	for i in range(0,len(lines)-1):
		for j in range(i+1, len(lines)):
			x, y = get_junction(lines[i], lines[j])
			if(x and y):
				intersections.append([x,y])
	return intersections

"""
Applies K-Means on the given list of 2D points. Returns n number of
cluster centers.
"""
def kmeans_centers(points, n):
	if len(points) < n:
		return None
	else:
		arr = np.asarray(points)
		kmeans = KMeans(n_clusters=n, random_state=0).fit(arr)
		return kmeans.cluster_centers_

"""
Weeds out the points that are out of the camera resolution.
"""
def get_noninf(points):
	noninfs = []
	for p in points:
		# This is bad and should be more modular
		if not (p[0] > 638 or p[0] < 0 or p[1] > 478 or p[1] <0):
			noninfs.append(p)
	return noninfs

"""
Calculates the corners for the biggest rectangle in the model image
This should be generalized in decide_points() function but this will 
do for now.

Or this could be called for near right angles, camera position exaclty
towards the middle of the field.

Returns top, bot : 2 arrays of size 2 thah each hold 2 points. 
Top Left > Top Right : Bottom Left > Bottom Right
"""
def get_model_corners(points):
	top = []
	bot = []
	sum_x = 0
	sum_y = 0
	for p in points:
		sum_x += p[0]
		sum_y += p[1]

	avg_x = sum_x / len(points)
	avg_y = sum_y / len(points)

	for p in points:
		if p[1] > avg_y:	# Y gets bigger at bottom pixels
			bot.append([int(p[0]),int(p[1])])
		else:
			top.append([int(p[0]),int(p[1])])

	if(len(top) >= 5 and len(bot) >= 5):
		top.sort(key = lambda x: x[0])
		bot.sort(key = lambda x: x[0])

		top_corner = [top[0], top[-1]]
		bot_corner = [bot[0], bot[-1]]

		return top_corner, bot_corner
	else:
		return [], []

"""
Calculate the euclidean distance between p1 and p2
"""
def distance(p1, p2):
	xdiff = p1[0] - p2[0]
	ydiff = p1[1] - p2[1]
	return math.sqrt(xdiff**2 + ydiff**2)

"""
Return top, bot : like get_model_lines but for 1-89 and 91-179 degree angles
"""
def get_frame_corners(points):
	top = [[],[]]
	bot = [[],[]]

	if points != []:
		p_min_x = min(points, key = lambda x: x[0])
		p_max_x = max(points, key = lambda x: x[0])
		p_min_y = min(points, key = lambda x: x[1])
		p_max_y = max(points, key = lambda x: x[1])

		top[0] = p_min_x
		top[1] = p_min_y
		bot[0] = p_max_y
		bot[1] = p_max_x

		d_diag = distance(top[0],bot[1])
		d_edge = distance(top[0],top[1])

		if d_edge == 0:
			d_edge = -1
		c = d_diag/d_edge
		print("D/E: ", c)
		if(c > 2):
			top[0] = p_min_y
			top[1] = p_max_x
			bot[0] = p_min_x
			bot[1] = p_max_y
		
		print("Points on top:", top)
		print("Points on bot:", bot)
		return top, bot
	else:
		return [], []
