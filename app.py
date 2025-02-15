import streamlit as st
import pickle
import numpy as np
import imageio
from skimage.transform import resize

# Muat model yang sudah disimpan
model = pickle.load(open('./model.p', 'rb'))

st.title("Deteksi Lokasi Parkir")
st.write("Upload gambar parkir untuk mengetahui apakah parkir kosong atau tidak.")

with st.form("upload_form"):
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Predict")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diupload", use_container_width=True)

    if submit:
        try:
            # Baca gambar menggunakan imageio
            image = imageio.imread(uploaded_file)
            
            # Resize gambar ke ukuran (15, 15) seperti saat training
            image_resized = resize(image, (15, 15))
            
            # Ubah gambar menjadi vektor (flatten)
            image_flat = image_resized.flatten().reshape(1, -1)
            
            # Lakukan prediksi
            prediction = model.predict(image_flat)
            label = "empty" if prediction[0] == 0 else "not_empty"
            
            st.success(f"Prediksi: {label}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")