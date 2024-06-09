import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import pillow_heif

# Fungsi untuk mengubah ukuran gambar
def resize_image(img, new_width, new_height):
    return cv2.resize(img, (new_width, new_height))

# Fungsi untuk mengekstraksi fitur dari gambar
def extract_features_from_image(img):
    new_width = 256
    new_height = 256
    resized_img = resize_image(img, new_width, new_height)
    
    # Pra-pemrosesan
    img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # Fitur warna dalam RGB
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    # Konversi ke HSV
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2HSV)
    
    hue_channel = hsv_img[:,:,0]
    saturation_channel = hsv_img[:,:,1]
    value_channel = hsv_img[:,:,2]

    hue_mean = np.mean(hue_channel)
    saturation_mean = np.mean(saturation_channel)
    value_mean = np.mean(value_channel)

    hue_std = np.std(hue_channel)
    saturation_std = np.std(saturation_channel)
    value_std = np.std(value_channel)

    # Buat vektor dari kombinasi fitur
    vector = [red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
              hue_mean, saturation_mean, value_mean, hue_std, saturation_std, value_std]

    return vector

def load_model():
    model = joblib.load('1.trained_svm_model.pkl')
    scaler = joblib.load('1.scaler.pkl')
    return model, scaler

def predict_image(img, model, scaler):
    features = extract_features_from_image(img)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction

def get_class_info(label):
    if label == 1:
        return "Segar"
    elif label == 2:
        return "Tidak Segar"
    elif label == 3:
        return "Busuk"
    else:
        return "Tidak diketahui"

# Memuat model SVM dan scaler
model, scaler = load_model()

# Judul aplikasi
st.markdown("<h1 style='text-align: center; color: #FFC3EE;'>Aplikasi Deteksi Kesegaran Ikan</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Aplikasi ini khusus untuk deteksi ikan nila merah, ikan kembung, dan ikan dencis.<br>Dapat digunakan pada ikan lain, tetapi tidak dapat dipastikan keakuratannya.<br>Selamat Mencoba!</p>", unsafe_allow_html=True)

# Pilihan fitur dengan select box
option = st.selectbox(
    "Pilih sumber gambar:",
    ('Kamera', 'Unggah Gambar')
)

# Kontainer utama untuk tata letak yang rapi
with st.container():
    if option == 'Kamera':
        if 'camera_opened' not in st.session_state:
            st.session_state.camera_opened = False

        if st.button('Buka Kamera', key='open_camera'):
            st.session_state.camera_opened = True

        if st.session_state.camera_opened:
            camera_image = st.camera_input("Ambil Foto")
            if camera_image is not None:
                img = Image.open(camera_image)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Konversi ke BGR untuk OpenCV
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Gambar dari Kamera', use_column_width=True)
                prediction = predict_image(img, model, scaler)
                class_info = get_class_info(prediction)
                st.success(f"Prediksi Ikan: {class_info}")
    
    elif option == 'Unggah Gambar':
        uploaded_img = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png", "heic"])
        if uploaded_img is not None:
            if uploaded_img.name.lower().endswith('.heic'):
                try:
                    heif_file = pillow_heif.read_heif(uploaded_img)
                    img = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                        heif_file.mode,
                        heif_file.stride,
                    )
                    img = np.array(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    st.error(f"Error: {e}")
                    img = None
            else:
                img = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), 1)
            
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Gambar yang diunggah', use_column_width=True)
                prediction = predict_image(img, model, scaler)
                class_info = get_class_info(prediction)
                st.success(f"Prediksi Ikan: {class_info}")
