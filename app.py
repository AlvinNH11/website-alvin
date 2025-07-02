import streamlit as st
import numpy as np
import io
import pickle
from PIL import Image
from tensorflow.keras.preprocessing import image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Klasifikasi Kerusakan Jalan",
    page_icon="🛣️",
    layout="centered"
)

# --- Konfigurasi Model & Kelas ---
MODEL_PATH = 'road_damage_detector_cnn_model.pkl'
ORIGINAL_CLASS_NAMES = ['Bagus', 'Memuaskan', 'Retak', 'Rusak']

# --- Fungsi-fungsi Inti (Caching untuk Performa) ---

@st.cache_resource
def load_model(model_path):
    """
    Memuat model machine learning dari file .pkl.
    Menggunakan cache agar model hanya dimuat sekali.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"[*] Model berhasil dimuat dari {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"Error: File model tidak ditemukan di '{model_path}'. Pastikan file model ada di folder yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

def model_predict(img_bytes, model_instance):
    """
    Fungsi untuk memproses gambar dan melakukan prediksi.
    Sama persis dengan logika di kode Flask Anda.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((150, 150))
    x = image.img_to_array(img)
    x = x / 255.0  # Normalisasi
    x = np.expand_dims(x, axis=0)
    prediction_array = model_instance.predict(x)
    return prediction_array

# --- Tampilan Antarmuka (UI) Aplikasi ---

# Judul dan deskripsi aplikasi
st.title("🛣️ Aplikasi Klasifikasi Kerusakan Jalan")
st.write(
    "Unggah gambar permukaan jalan untuk diprediksi tingkat kerusakannya. "
    "Model akan mengklasifikasikan gambar ke dalam kategori: Bagus, Memuaskan, Retak, atau Rusak."
)

# Memuat model
model = load_model(MODEL_PATH)

# Hanya melanjutkan jika model berhasil dimuat
if model is not None:
    # Komponen untuk mengunggah file gambar
    uploaded_file = st.file_uploader(
        "Pilih gambar jalan...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah oleh pengguna
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)
        
        # Tombol untuk memicu proses klasifikasi
        if st.button("Klasifikasikan Gambar"):
            with st.spinner("Sedang menganalisis..."):
                try:
                    # Membaca file sebagai bytes
                    img_bytes = uploaded_file.getvalue()
                    
                    # Melakukan prediksi
                    preds = model_predict(img_bytes, model)
                    
                    # Mendapatkan hasil
                    predicted_index = np.argmax(preds[0])
                    predicted_class = ORIGINAL_CLASS_NAMES[predicted_index]
                    confidence = np.max(preds[0]) * 100
                    
                    # Menampilkan hasil prediksi
                    st.success(f"**Hasil Prediksi: {predicted_class}**")
                    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")
                    
                    # Menampilkan detail probabilitas (opsional)
                    with st.expander("Lihat Detail Probabilitas"):
                        prob_data = {
                            "Kelas": ORIGINAL_CLASS_NAMES,
                            "Probabilitas (%)": [f"{p*100:.2f}" for p in preds[0]]
                        }
                        st.table(prob_data)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# Footer
st.markdown("---")
st.markdown("Dibuat dengan [Streamlit](https://streamlit.io)")
