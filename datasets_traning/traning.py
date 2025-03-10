import cv2
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA


# Function to compute orientation angle using PCA
def compute_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Combine all contours into one array of points
    points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])

    # Apply PCA to find the orientation
    pca = PCA(n_components=2)
    pca.fit(points)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0]) * 180 / np.pi

    return angle


# Directory containing images
image_dir = r"G:\capture_images"  # Ensure correct raw string for Windows path
angles = []
image_names = []

# Loop through images in the dataset
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    image = cv2.imread(img_path)

    if image is not None:
        angle = compute_angle(image)
        if angle is not None:
            angles.append(angle)
            image_names.append(filename)

# Compute mean angle and save to CSV
if angles:
    mean_angle = np.mean(angles)
    print(f"Mean Angle: {mean_angle:.2f} degrees")

    # Create a DataFrame
    df = pd.DataFrame({"Image Name": image_names, "Angle (degrees)": angles})

    # Add mean angle as a separate row
    mean_data = pd.DataFrame({"Image Name": ["Mean Angle"], "Angle (degrees)": [mean_angle]})
    df = pd.concat([df, mean_data], ignore_index=True)

    # Save to CSV
    csv_path = r"G:\capture_images\angle_data1.csv"  # Change the path if needed
    df.to_csv(csv_path, index=False)

    print(f"Data saved successfully to {csv_path}")

else:
    print("No valid images found for analysis.")
