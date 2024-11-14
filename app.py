import os
import cv2
import numpy as np
import pandas as pd
import pickle  
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops
import streamlit as st
from sklearn.model_selection import train_test_split


main_folder_path = 'D:\streamlit_project\data'  
model_pickle_path = 'model_random_forest.pkl'   

def extract_glcm_features(image):
    citra_resize = cv2.resize(image, (256, 256))
    citra_hsv = cv2.cvtColor(citra_resize, cv2.COLOR_RGB2HSV)
    kanal_saturasi = citra_hsv[:, :, 1]

    _, mask = cv2.threshold(kanal_saturasi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    return citra_resize, masked_grayscale, inverted_mask_glcm, [contrast, correlation, energy, homogeneity]

def extract_morphological_features(image):
    citra_resize = cv2.resize(image, (256, 256))
    citra_hsv = cv2.cvtColor(citra_resize, cv2.COLOR_RGB2HSV)
    kanal_saturasi = citra_hsv[:, :, 1]

    _, mask = cv2.threshold(kanal_saturasi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask_opening_citra = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closing_citra = cv2.morphologyEx(mask_opening_citra, cv2.MORPH_CLOSE, kernel)

    inverted_mask_morfologi = cv2.bitwise_not(mask_closing_citra)

    
    contours, _ = cv2.findContours(inverted_mask_morfologi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0, 0, 0  

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    moments = cv2.moments(cnt)
    eccentricity = 0 if (moments["mu20"] + moments["mu02"]) == 0 else (
        ((moments["mu20"] - moments["mu02"]) ** 2 + 4 * moments["mu11"] ** 2) ** 0.5 /
        (moments["mu20"] + moments["mu02"])
    )

    return inverted_mask_morfologi, area, perimeter, eccentricity


def process_image(image_path):
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        return None  

    
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    citra_resized, masked_grayscale, inverted_mask_glcm, glcm_features = extract_glcm_features(img_rgb)
    inverted_mask_morfologi, area, perimeter, eccentricity = extract_morphological_features(img_rgb)
    
    return img_rgb, citra_resized, masked_grayscale, inverted_mask_morfologi, glcm_features + [area, perimeter, eccentricity]  # Kembalikan citra resized dan morph filled


def process_folder(folder_path):
    features = []
    labels = []
    for subdir, dirs, files in os.walk(folder_path):
        label = os.path.basename(subdir)
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith(('.png', '.jpg', '.jpeg')):
                _, citra_resized, masked_grayscale, inverted_mask_morfologi, feature_data = process_image(file_path)
                if feature_data is not None:  
                    features.append(feature_data)
                    labels.append(label)
    return features, labels


def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    st.success(f"Model berhasil disimpan ke {file_path}")


def load_model(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        return None


model_rf = load_model(model_pickle_path) 
if model_rf is None:
    st.warning(f"Tidak ditemukan model di {model_pickle_path}. Melatih model baru...")

    
    features, labels = process_folder(main_folder_path)
    if len(features) == 0:
        st.error("Tidak ada fitur yang dapat diekstraksi. Pastikan gambar dalam folder benar.")
    else:
        df = pd.DataFrame(features, columns=['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Area', 'Perimeter', 'Eccentricity'])
        df['Label'] = labels

        X = df.drop('Label', axis=1)
        y = df['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Latih model Random Forest
        model_rf = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=42)
        model_rf.fit(X_train, y_train)

        # Simpan model ke file pickle
        save_model(model_rf, model_pickle_path)


st.title("Klasifikasi Penyakit Pada Daun Mangga Berdasarkan Citra Penyakit Daun Mangga Menggunakan Random Forest")
st.write("Silahkan unggah citra penyakit daun mangga :")


uploaded_file = st.file_uploader("Silahkan memilih citra", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_rgb = cv2.imdecode(file_bytes, 1)

    
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    
    citra_resized, masked_grayscale, inverted_mask_glcm, features_upload = extract_glcm_features(img_rgb)
    inverted_mask_morfologi, area, perimeter, eccentricity = extract_morphological_features(img_rgb)
    features_upload += [area, perimeter, eccentricity]

    
    st.write("### Citra Resize, Citra Grayscale, dan Citra Morfologi")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(citra_resized, caption='Citra Hasil Resize', use_column_width=True)

    with col2:
        st.image(masked_grayscale, caption='Citra Hasil Grayscale', use_column_width=True)

    with col3:
        st.image(inverted_mask_morfologi, caption='Citra Hasil Morfologi', use_column_width=True)

    if features_upload:  
        
        prediction = model_rf.predict([features_upload])

        
        st.write(f"### Hasil Klasifikasi: {prediction[0]}")

       
        st.write("### Nilai Ekstraksi Fitur")
        feature_names = ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Area', 'Perimeter', 'Eccentricity']
        features_df = pd.DataFrame([features_upload], columns=feature_names)
        st.table(features_df)
    else:
        st.error("Gagal mengekstrak fitur dari gambar yang diunggah.")
