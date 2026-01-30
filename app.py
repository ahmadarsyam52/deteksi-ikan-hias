import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Konfigurasi Tampilan
st.set_page_config(page_title="Deteksi Ikan Hias UAS", page_icon="🐟")
st.title("🐟 Deteksi Jenis Ikan Hias")
st.write("Aplikasi ini dapat mendeteksi **Ikan Koki, Cupang, dan Mutiara**.")

# 1. Load Model (Memanggil file best.pt yang sudah kamu download)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 2. Upload Gambar
uploaded_file = st.file_uploader("Unggah foto ikan...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Membuka gambar yang diupload
    image = Image.open(uploaded_file)
    
    # Menampilkan tombol proses
    if st.button('Mulai Deteksi'):
        # 3. Prediksi menggunakan model YOLO
        results = model(image)
        
        # 4. Visualisasi Hasil
        res_plotted = results[0].plot()
        
        # Tampilkan Hasil
        st.image(res_plotted, caption='Hasil Deteksi AI', use_container_width=True)
        
        # Tampilkan Detail Probabilitas
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                st.success(f"Terdeteksi: **{label}** (Tingkat Keyakinan: {conf:.2%})")