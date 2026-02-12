import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Jaya Jaya Institut - Retention System",
    page_icon="🎓",
    layout="wide"
)

# Load Model dan Assets dengan Caching
@st.cache_resource
def load_assets():
    try:
        data = joblib.load('dropout_prediction_model.joblib')
        return data
    except FileNotFoundError:
        return None

data = load_assets()

if data is None:
    st.error("⚠️ File 'dropout_prediction_model.joblib' tidak ditemukan!")
    st.stop()

model = data['model']
scaler = data['scaler']
expected_cols = data['features']
numeric_cols = data['numeric_cols']
defaults = data['defaults']
target_names = data['target_names']

# Header
st.title("🎓 Sistem Deteksi Dini Mahasiswa Dropout")
st.markdown("---")

# Layout
# Kita bagi halaman menjadi 2 Tab
tab1, tab2 = st.tabs(["🔮 Prediksi", "ℹ️ Penjelasan Fitur & Range Nilai"])

# Form input user
with st.form("prediction_form"):
    
    # Data history pendidikan terakhir (Top Feature #4, #5)
    st.subheader("🏫 Riwayat Pendidikan (History)")
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        # Range nilai biasanya 0-200 untuk dataset ini (Portugal scale)
        # Kita ambil default dari median training data
        adm_grade = st.number_input("Nilai Ujian Masuk (Admission Grade)", 
                                    min_value=0.0, max_value=200.0, 
                                    value=float(defaults.get('Admission_grade', 120.0)))
                                    
    with col_hist2:
        prev_grade = st.number_input("Nilai Jenjang Sebelumnya (Previous Qualification)", 
                                     min_value=0.0, max_value=200.0, 
                                     value=float(defaults.get('Previous_qualification_grade', 130.0)))
    
    st.markdown("---")

    # Data Akademik Semester 1 & 2 (Top Feature #1, #2, #3)
    st.subheader("📊 Performa Akademik (Semester 1 & 2)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Semester 1**")
        sem1_units = st.number_input("SKS Diambil (Sem 1)", min_value=0, value=20)
        sem1_appr = st.number_input("SKS Lulus (Sem 1)", min_value=0, value=18)
        sem1_grade = st.number_input("Nilai Rata-rata (Sem 1)", min_value=0.0, max_value=20.0, value=14.0)
        
    with col2:
        st.markdown("**Semester 2**")
        sem2_units = st.number_input("SKS Diambil (Sem 2)", min_value=0, value=20)
        sem2_appr = st.number_input("SKS Lulus (Sem 2)", min_value=0, value=18)
        sem2_grade = st.number_input("Nilai Rata-rata (Sem 2)", min_value=0.0, max_value=20.0, value=14.0)

    st.markdown("---")

    # Kelompok 2: Data Ekonomi & Demografi (Top Feature #8)
    st.subheader("👤 Data Pribadi & Ekonomi")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        tuition = st.selectbox("Status SPP", options=[1, 0], format_func=lambda x: "Lancar (Up to date)" if x == 1 else "Menunggak")
        scholarship = st.selectbox("Penerima Beasiswa?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        debtor = st.selectbox("Memiliki Utang?", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")

    with col4:
        gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
        marital = st.selectbox("Status Pernikahan", options=[1, 2, 3, 4, 5, 6], 
                               format_func=lambda x: {1:"Lajang", 2:"Menikah", 3:"Duda/Janda", 4:"Facto", 5:"Pisah", 6:"Lainnya"}.get(x, x))
        
    with col5:
        age = st.number_input("Usia saat Mendaftar", min_value=17, max_value=70, value=20)
        displaced = st.selectbox("Status Rantau (Displaced)", options=[1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")

    # Tombol Submit
    submitted = st.form_submit_button("🔍 Prediksi Status")

# Prediction Pipeline
if submitted:
    # Kumpulkan Input User ke dalam Dictionary
    user_input = {
        # Fitur
        'Admission_grade': adm_grade,
        'Previous_qualification_grade': prev_grade,
        'Curricular_units_1st_sem_enrolled': sem1_units,
        'Curricular_units_1st_sem_approved': sem1_appr,
        'Curricular_units_1st_sem_grade': sem1_grade,
        'Curricular_units_2nd_sem_enrolled': sem2_units,
        'Curricular_units_2nd_sem_approved': sem2_appr,
        'Curricular_units_2nd_sem_grade': sem2_grade,
        'Age_at_enrollment': age,
        'Tuition_fees_up_to_date': tuition,
        'Scholarship_holder': scholarship,
        'Debtor': debtor,
        'Gender': gender,
        'Displaced': displaced,
        'Marital_status': marital
    }
    
    # Gabungkan dengan Defaults
    # Note: Defaults disini berisi nilai 0 (scaled median), ini aman untuk fitur yang tidak penting (GDP dll)
    # karena nanti scaler akan menganggapnya sebagai "rata-rata".
    full_data = defaults.copy()
    full_data.update(user_input)
    
    # Buat DataFrame
    df = pd.DataFrame([full_data])
    
    # Scaling
    try:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    except Exception as e:
        st.error(f"Error Scaling: {e}")
        st.stop()

    # Encoding & Reindex
    df_ohe = pd.get_dummies(df)
    df_final = df_ohe.reindex(columns=expected_cols, fill_value=0)

    # Prediksi
    prediction = model.predict(df_final)[0]
    proba = model.predict_proba(df_final)[0]
    result_label = target_names[prediction]
    
    # Tampilkan Hasil
    st.markdown("---")
    col_res, col_metrics = st.columns([1, 1])
    
    with col_res:
        if result_label == "Dropout":
            st.error(f"### ⚠️ Prediksi: DROPOUT")
            st.write("Mahasiswa berisiko tinggi. Segera lakukan intervensi.")
        elif result_label == "Enrolled":
            st.warning(f"### 🎓 Prediksi: ENROLLED")
            st.write("Mahasiswa masih aman, pantau terus.")
        else:
            st.success(f"### 🎉 Prediksi: GRADUATE")
            st.write("Performa akademik sangat baik.")

    with col_metrics:
        st.caption("Probabilitas:")
        st.progress(proba[0], text=f"Dropout: {proba[0]:.2%}")
        st.progress(proba[2], text=f"Graduate: {proba[2]:.2%}")

# Penjelasan & Bantuan
with tab2:
    st.header("📖 Kamus Data & Range Nilai")
    st.markdown("""
    Dataset ini menggunakan standar akademik Portugal yang mungkin berbeda dengan Indonesia. 
    Berikut adalah panduan pengisian form:
    """)

    # Membuat Data Frame untuk Penjelasan agar rapi
    help_data = {
        "Nama Fitur": [
            "Admission Grade", 
            "Previous Qualification Grade", 
            "Curricular Units (SKS)", 
            "Semester Grade (Nilai)",
            "Tuition Fees",
            "Displaced"
        ],
        "Deskripsi": [
            "Nilai ujian masuk perguruan tinggi.", 
            "Nilai akhir dari jenjang pendidikan sebelumnya (SMA/D3).", 
            "Jumlah mata kuliah (SKS) yang diambil atau lulus.", 
            "Nilai rata-rata IP semester.",
            "Status pembayaran uang kuliah.",
            "Apakah mahasiswa tinggal jauh dari rumah (rantau)?"
        ],
        "Range / Skala Nilai": [
            "0 - 200 (Contoh: 140.5)", 
            "0 - 200 (Contoh: 130.0)", 
            "0 - 20 (Integer)", 
            "0 - 20 (Contoh: 14.5)",
            "Lancar / Menunggak",
            "Ya / Tidak"
        ]
    }
    
    st.table(pd.DataFrame(help_data))