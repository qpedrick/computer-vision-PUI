import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# image = cv2.imread('./calibration_far_09-22-24.jpg')
image = cv2.imread('./calibration_close_09-22-24.jpg')

if image is None:
    print("Error: Image not found or unable to load.")
else:
    print("Image loaded successfully.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the original grayscale image
    gray_preprocessed = gray

    # Visualize the preprocessed image
    plt.imshow(gray_preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    plt.show()

    # Define internal corners (width boxes - 1, height boxes - 1)
    checkerboard_size = (5, 9)

    # Define flags
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS


    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_preprocessed, checkerboard_size, flags)

    if ret:
        print("Checkerboard detected.")

        # Refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray_preprocessed, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        image_corners = cv2.drawChessboardCorners(image.copy(), checkerboard_size, corners_refined, ret)
        plt.imshow(cv2.cvtColor(image_corners, cv2.COLOR_BGR2RGB))
        plt.title('Detected Chessboard Corners')
        plt.axis('off')
        plt.show()
    else:
        print("Checkerboard not found.")
        # Optionally, display the attempted detection
        image_attempt = cv2.drawChessboardCorners(image.copy(), checkerboard_size, corners, ret)
        plt.imshow(cv2.cvtColor(image_attempt, cv2.COLOR_BGR2RGB))
        plt.title('Detection Attempt')
        plt.axis('off')
        plt.show()
