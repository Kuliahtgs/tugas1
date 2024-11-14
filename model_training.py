import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt  # Tambahkan import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops

# Path to the main folder (update to your local path)
main_folder_path = 'D:\streamlit_project\data'

# Function to extract GLCM features
def extract_glcm_features(image):
    citra_resize = cv2.resize(image, (256, 256))
    citra_hsv = cv2.cvtColor(citra_resize, cv2.COLOR_RGB2HSV)
    saturasi_channel = citra_hsv[:, :, 1]

    _, mask = cv2.threshold(saturasi_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    citra_grayscale = cv2.cvtColor(citra_resize, cv2.COLOR_RGB2GRAY)
    inverted_mask_glcm = cv2.bitwise_not(mask)
    masked_grayscale = cv2.bitwise_and(citra_grayscale, citra_grayscale, mask=inverted_mask_glcm)

    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_matrix = graycomatrix(masked_grayscale, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm_matrix, 'contrast').mean()
    correlation = graycoprops(glcm_matrix, 'correlation').mean()
    energy = graycoprops(glcm_matrix, 'energy').mean()
    homogeneity = graycoprops(glcm_matrix, 'homogeneity').mean()

    return [contrast, correlation, energy, homogeneity]

# Function to extract morphological features
def extract_morphological_features(image):
    citra_resize = cv2.resize(image, (256, 256))
    citra_hsv = cv2.cvtColor(citra_resize, cv2.COLOR_RGB2HSV)
    saturasi_channel = citra_hsv[:, :, 1]

    _, mask = cv2.threshold(saturasi_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask_opening_citra = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closing_citra = cv2.morphologyEx(mask_opening_citra, cv2.MORPH_CLOSE, kernel)

    inverted_mask_morfologi = cv2.bitwise_not(mask_closing_citra)

    # Calculate area, perimeter, and eccentricity from contours
    contours, _ = cv2.findContours(inverted_mask_morfologi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0, 0, 0  # Return 0 if no contours are found

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    moments = cv2.moments(cnt)
    eccentricity = 0 if (moments["mu20"] + moments["mu02"]) == 0 else (
        ((moments["mu20"] - moments["mu02"]) ** 2 + 4 * moments["mu11"] ** 2) ** 0.5 /
        (moments["mu20"] + moments["mu02"])
    )

    return [area, perimeter, eccentricity]

# Function to process the image and extract features
def process_and_extract_features(image_path, label):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Extract GLCM features
    glcm_features = extract_glcm_features(img_rgb)

    # Extract morphological features
    morfologi_features = extract_morphological_features(img_rgb)

    # Combine features
    combined_features = glcm_features + morfologi_features

    return combined_features, label

# Function to process images from each subfolder
def process_folders(folder_path):
    features = []
    labels = []

    for subdir, dirs, files in os.walk(folder_path):
        if not files:
            continue
        label = os.path.basename(subdir)  # Use the subfolder name as the label
        for image_file in files:
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, image_file)
                combined_features, label = process_and_extract_features(image_path, label)
                features.append(combined_features)
                labels.append(label)

    return np.array(features), np.array(labels)

# Process images from all subfolders and extract features
features, labels = process_folders(main_folder_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Display classification report
print(classification_report(y_test, y_pred))