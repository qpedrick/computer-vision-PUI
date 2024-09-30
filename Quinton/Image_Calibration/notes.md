# Image Calibration

## Road Blocks & Solutions

- Need multiple images of the calibration board from different angles
    - And/or model # of the gopro to search for calibration coefficients online

## Camera Calibration Coefficients

1. Camera Matrix (K) : Contains the intrinsic parameters of the camera, including the focal lengths and optical centers.
2. Distortion Coefficients (d) : Represent the lens distortion, including radial and tangential distortion.
3. Rotation Vectors (rvecs) and Translation Vectors (tvecs) : Describe the camera's position and orientation (extrinsic parameters) relative to the calibration pattern.

**Can find these parameters online based on the camera sometimes which avoids having to have 10-20 images of the board from different angles*