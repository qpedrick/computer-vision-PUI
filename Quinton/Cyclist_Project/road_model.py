import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Parameters for the road
A = 5       # Amplitude of the sine wave (width of the curves)
B = 0.05    # Frequency of the sine wave (how often it curves)
t = np.linspace(0, 100, 500)  # Parameter t from 0 to 100 with 500 points

# Parametric equations for the road
x = t
y = A * np.sin(B * t)
z = np.zeros_like(t)  # z = 0 for all points since the road is on the z-plane

# Cyclist position (near the beginning of the curve)
t_cyclist = 10  # Parameter value for cyclist's position
x_cyclist = t_cyclist
y_cyclist = A * np.sin(B * t_cyclist)
z_cyclist = 0  # On the road surface

# Obstacle position (50-75 units ahead along the curve)
delta_t = 60  # Distance ahead along the parameter t
t_obstacle = t_cyclist + delta_t
x_obstacle = t_obstacle
y_obstacle = A * np.sin(B * t_obstacle)
z_obstacle = 0  # On the road surface

# Drone position (ahead of both cyclist and obstacle)
t_drone = t_obstacle + 20  # Ahead of both
x_drone = t_drone
y_drone = A * np.sin(B * t_drone)
z_drone = 75  # Altitude of the drone

# Function to compute the differential arc length
def arc_length_integrand(t):
    dx_dt = 1  # Derivative of x(t) = t with respect to t
    dy_dt = A * B * np.cos(B * t)  # Derivative of y(t) with respect to t
    dz_dt = 0  # z(t) is constant at 0
    return np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

# Calculate the arc length between cyclist and obstacle
path_distance, _ = quad(arc_length_integrand, t_cyclist, t_obstacle)
print(f"Path distance between cyclist and obstacle: {path_distance:.2f} units")

# Angles from drone to cyclist
dx_cyclist = x_cyclist - x_drone
dy_cyclist = y_cyclist - y_drone
dz_cyclist = z_cyclist - z_drone

distance_cyclist = np.sqrt(dx_cyclist**2 + dy_cyclist**2 + dz_cyclist**2)
theta_cyclist = np.degrees(np.arctan2(dy_cyclist, dx_cyclist))
phi_cyclist = np.degrees(np.arcsin(dz_cyclist / distance_cyclist))

# Angles from drone to obstacle
dx_obstacle = x_obstacle - x_drone
dy_obstacle = y_obstacle - y_drone
dz_obstacle = z_obstacle - z_drone

distance_obstacle = np.sqrt(dx_obstacle**2 + dy_obstacle**2 + dz_obstacle**2)
theta_obstacle = np.degrees(np.arctan2(dy_obstacle, dx_obstacle))
phi_obstacle = np.degrees(np.arcsin(dz_obstacle / distance_obstacle))

print(f"\nDrone to Cyclist:")
print(f"  Distance: {distance_cyclist:.2f} units")
print(f"  Azimuth Angle (theta): {theta_cyclist:.2f} degrees")
print(f"  Elevation Angle (phi): {phi_cyclist:.2f} degrees")

print(f"\nDrone to Obstacle:")
print(f"  Distance: {distance_obstacle:.2f} units")
print(f"  Azimuth Angle (theta): {theta_obstacle:.2f} degrees")
print(f"  Elevation Angle (phi): {phi_obstacle:.2f} degrees")

# Plotting the road
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='gray', linewidth=2, label='Road')

# Plotting the cyclist
ax.scatter(x_cyclist, y_cyclist, z_cyclist, color='blue', s=100, label='Cyclist')

# Plotting the obstacle
ax.scatter(x_obstacle, y_obstacle, z_obstacle, color='red', s=100, label='Obstacle')

# Plotting the drone
ax.scatter(x_drone, y_drone, z_drone, color='green', s=100, marker='^', label='Drone')

# Setting labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Model with Cyclist, Obstacle, and Drone')

# Set equal aspect ratio for all axes
ax.set_box_aspect([np.ptp(a) for a in [x, y, np.append(z, z_drone)]])

# Show the legend
ax.legend()

# Display the plot
plt.show()