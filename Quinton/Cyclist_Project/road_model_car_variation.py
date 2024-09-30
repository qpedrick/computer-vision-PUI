import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Parameters for the road
A = 10      # Increased amplitude for a more pronounced curve
B = 0.1     # Increased frequency
t = np.linspace(0, 100, 1000)  # More points for smoother curve

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

# Observer car position (ahead of both cyclist and obstacle)
t_car = t_obstacle + 20  # Ahead of both
x_car = t_car
y_car = A * np.sin(B * t_car)
z_car = 0  # On the road surface

# Camera height above the car
z_camera = z_car + 2.13  # Cameras mounted 2.13 meters above the road (7 feet)

# Function to compute the differential arc length
def arc_length_integrand(t):
    dx_dt = 1  # Derivative of x(t) = t with respect to t
    dy_dt = A * B * np.cos(B * t)  # Derivative of y(t) with respect to t
    dz_dt = 0  # z(t) is constant at 0
    return np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

# Calculate the arc length between cyclist and obstacle
path_distance_cyclist_obstacle, _ = quad(arc_length_integrand, t_cyclist, t_obstacle)

# Calculate the arc length between cyclist and observer car
path_distance_cyclist_car, _ = quad(arc_length_integrand, t_cyclist, t_car)

print(f"Path distance between cyclist and obstacle: {path_distance_cyclist_obstacle:.2f} meters")
print(f"Path distance between cyclist and observer car: {path_distance_cyclist_car:.2f} meters")

# Plotting the road
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='gray', linewidth=2, label='Road')

# Plotting the cyclist
ax.scatter(x_cyclist, y_cyclist, z_cyclist, color='blue', s=100, label='Cyclist')

# Plotting the obstacle
ax.scatter(x_obstacle, y_obstacle, z_obstacle, color='red', s=100, label='Obstacle')

# Plotting the observer car
ax.scatter(x_car, y_car, z_car, color='green', s=100, label='Observer Car')

# Plotting the cameras on top of the car
ax.scatter(x_car, y_car, z_camera, color='green', s=100, marker='^', label='Cameras')

# Setting labels and title
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('3D Model with Cyclist, Obstacle, and Observer Car with Cameras')

# Set equal aspect ratio for all axes
ax.set_box_aspect([np.ptp(a) for a in [x, y, np.append(z, z_camera)]])

# Show the legend
ax.legend()

# Display the plot
plt.show()