import cv2 as cv
from utils import *


# Window adjustment==========================
w_width  = 540
w_height = 400

cv.namedWindow("Raw",cv.WINDOW_NORMAL)
cv.namedWindow("Model",cv.WINDOW_NORMAL)
cv.namedWindow("Hough",cv.WINDOW_NORMAL)

cv.resizeWindow("Raw", w_width, w_height)
cv.resizeWindow("Model", w_width, w_height)
cv.resizeWindow("Hough", w_width, w_height)

cv.moveWindow("Hough", w_width, w_height);
cv.moveWindow("Model", 0, w_height);
cv.moveWindow("Raw", w_width, 0);
# Window adjustment==========================

# Parameters=================================
# Hough Lines
h_rho = 0.8
h_theta = np.pi / 180
h_tresh =70 	# Minimum number of pixed in a line
h_lines = None
h_srn = 0 	# divisor for rho	
h_stn = 0  	# divisor for theta

# Canny
c_t1 = 50
c_t2 = c_t1*3
# Parameters=================================
#Process model image=========================
model_img = cv.imread("model.jpg")
model_gray = cv.cvtColor(model_img, cv.COLOR_BGR2GRAY)    
model_canny = cv.Canny(model_gray, c_t1, c_t2)
model_lines = cv.cvtColor(model_canny, cv.COLOR_GRAY2BGR)
source_lines = cv.HoughLines(model_canny, 1.1, h_theta, 247, h_lines, h_srn, h_stn)

# Add the lines that we got from Hough routine to the model_lines frame
# Also convert polar coordinates to euclidean coordinats for k-means clustering
cartesian_lines = []
if source_lines is not None:
	for i in range(0, len(source_lines)):
		pt1, pt2 = add_line(source_lines, i, model_lines)
		cartesian_lines.append([[pt1[0],pt1[1]], [pt2[0],pt2[1]]])

# Find all junctions and perform k-means
# K-means cluster size and hough parameters are decided via
# Trial and error and explained in the project report
all_intersections = junctions(cartesian_lines)
cluster_centers = kmeans_centers(all_intersections , 10)

# Mark cluster centers on the modal lines
mark_all(cluster_centers, model_lines)

# Write the image to disk to show on the report :)
# cv.imwrite("model_lines.png", model_lines)

# Calculate corners
m_top,m_bot = get_model_corners(cluster_centers)
print("Cluster centers:", cluster_centers)
print("Points on top:", m_top)
print("Points on bot:", m_bot)
model_joined_topbot = m_top + m_bot
#Process model image=========================

# Initilize warped image incase the program fails to calculate one
# at the first frame
warped_img = []
cap = cv.VideoCapture(0)

while(True):
	_ , frame_raw = cap.read()

	# Raw frame > gray scaled > Canny > Hough Lines====
	frame_gray = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)    
	frame_canny = cv.Canny(frame_gray, c_t1, c_t2)
	frame_lines = cv.cvtColor(frame_canny, cv.COLOR_GRAY2BGR)
	lines = cv.HoughLines(frame_canny, h_rho, h_theta, h_tresh, h_lines, h_srn, h_stn)
	# =================================================
	
	cartesian_lines = []
	if lines is not None:
		for i in range(0, len(lines)):
			pt1, pt2 = add_line(lines, i, frame_lines)
			cartesian_lines.append([[pt1[0],pt1[1]], [pt2[0],pt2[1]]])
 
	# Find all junctions and perform k-means and mark them
	junction_points = junctions(cartesian_lines)
	cluster_centers = kmeans_centers(junction_points, 37)
	mark_all(junction_points, frame_lines, rbg=(0,0,255))

	# Check if the clusters center calculation is insufficent
	if type(cluster_centers) is np.ndarray:
		n_noninf = get_noninf(cluster_centers)
		
		# 4 is number of corners, in future this should be generalized 
		# to n points homography
		if(len(n_noninf) >= 4):
			top, bot = get_frame_corners(n_noninf)
			frame_joined_topbot = top + bot
			if(len(frame_joined_topbot) == 4):
				# Calculate the mask for the warping
				mask = np.array([])
				mask , status = cv.findHomography(np.array(frame_joined_topbot), np.array(model_joined_topbot)) 
				print("Mask:\n", mask)
				# Get frame resolution for cv.warpPerspective()
				height, width, channels = model_img.shape
				warped_img = cv.warpPerspective(frame_raw, mask, (width, height))
		# Mark lines on the canny+hough frame
		mark_all(n_noninf, frame_lines)
		print("Clusters: ", len(n_noninf)) # Number of cluster centers on screen


	# Display the frames
	cv.imshow("Raw", frame_raw)
	cv.imshow("Hough", frame_lines)
	# If warped image is calculated(Can be an old frame if the currents couldn't be calculated)
	if (len(warped_img) > 0): 
		cv.imshow("Model", warped_img)

	# 1000/50 = 20 FPS, Quit on keypress Q
	if cv.waitKey(50) & 0xFF == ord('q'): 
		break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()