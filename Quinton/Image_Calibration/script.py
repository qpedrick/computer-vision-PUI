import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Define the dimensions of the checkerboard
checkerboard_size = (5, 9)  # (width, height) internal corners
square_size = 10  # Size of a square in centimeters

# Prepare object points based on the real-world dimensions of the checkerboard
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Multiply by square size to get actual coordinates in cm

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# List of images to use for calibration
images = ['./calibration_close_09-22-24.jpg', './calibration_far_09-22-24.jpg']  # Replace with your actual image filenames

for fname in images:
    # Load the image
    image = cv2.imread(fname)
    
    if image is None:
        print(f"Error: Image {fname} not found or unable to load.")
        continue
    else:
        print(f"Processing image {fname}...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if ret:
        print("Checkerboard detected.")
        
        # Refine the corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Append object points and image points
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        
        # Draw and display the corners
        image_corners = cv2.drawChessboardCorners(image.copy(), checkerboard_size, corners_refined, ret)
        plt.imshow(cv2.cvtColor(image_corners, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Corners in {fname}')
        plt.axis('off')
        plt.show()
    else:
        print(f"Checkerboard not found in image {fname}.")

# Perform camera calibration using the collected object points and image points
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Camera calibration successful.")
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
else:
    print("Camera calibration failed.")

# Save the calibration results for later use
# np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# print("\nCalibration results saved to 'camera_calibration.npz'.")

# Test Case - Will need more camera calibration images
# Load an image to undistort
# img = cv2.imread(images[1])
# h, w = img.shape[:2]

# # Get the optimal new camera matrix
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
#     camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# # Undistort the image
# undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# # Crop the image (optional, in case there's black edges)
# # x, y, w, h = roi
# # undistorted_img = undistorted_img[y:y+h, x:x+w]

# # Display the original and undistorted images
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
# plt.title('Undistorted Image')
# plt.axis('off')

# plt.show()
