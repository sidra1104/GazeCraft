import cv2
import numpy as np
import json

def load_calibration_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_accuracy(calibration_circle_pos, homography, calibration_eye_pos):
    """Calculates the accuracy between fixed calibration positions and screen points after homography."""
    screen_points = []
    
    for eyes_pos in calibration_eye_pos:
        eyes_point = np.array([[eyes_pos[0], eyes_pos[1], 1]]).T  # Convert to homogeneous coordinates
        transformed_point = np.dot(homography, eyes_point)
        screen_point = [transformed_point[0] / transformed_point[2], transformed_point[1] / transformed_point[2]]
        screen_points.append(screen_point)
    
    screen_points = np.array(screen_points).reshape(-1, 2)
    calibration_circle_pos = np.array(calibration_circle_pos)
    
    distances = np.linalg.norm(calibration_circle_pos - screen_points, axis=1)
    mean_error = np.mean(distances)
    
    return screen_points, distances, mean_error

# Load data from JSON file
data = load_calibration_data('calibration_data.json')

calibration_circle_pos = data["calibration_circle_pos"]
calibration_eye_pos = data["calibration_eye_pos"]
homography_matrix = np.array(data["homography"])

screen_points, distances, mean_error = calculate_accuracy(calibration_circle_pos, homography_matrix, calibration_eye_pos)

print("Calculated Screen Points:")
print(screen_points,'\n')

for i, d in enumerate(distances):
    print(f"Point {i+1}: Distance = {d:.2f}")

print(f"\n\nMean Error: {mean_error:.2f}")
