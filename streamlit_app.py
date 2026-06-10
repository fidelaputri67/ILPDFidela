import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="ILPD Liver Disease App",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD MODEL (Naive Bayes)
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load("ilpd_best_model.pkl")
    return model

model = load_model()

# ==========================================
# NAVIGATION
# ==========================================
menu = st.sidebar.selectbox(
    "📌 Menu",
    ["Home", "Dataset ILPD", "Prediksi", "Kesehatan Hati"]
)

# ==========================================
# RANGE NORMAL
# ==========================================
normal_range = {
    "Total Bilirubin": "0.1 - 1.2 mg/dL",
    "Direct Bilirubin": "0.0 - 0.3 mg/dL",
    "ALT": "7 - 56 U/L",
    "AST": "10 - 40 U/L",
    "Total Proteins": "6.0 - 8.3 g/dL",
    "Albumin": "3.5 - 5.0 g/dL"
}

# ==========================================
# HOME
# ==========================================
if menu == "Home":
    st.title("🩺 Sistem Prediksi Penyakit Hati (ILPD)")
    st.info("Model Naive Bayes untuk klasifikasi penyakit hati")

    st.markdown("""
    ### 📌 Fitur aplikasi:
    - Prediksi penyakit hati
    - Informasi dataset ILPD
    - Edukasi kesehatan hati
    """)

# ==========================================
# DATASET
# ==========================================
elif menu == "Dataset ILPD":

    st.title("📊 Dataset ILPD")

    st.markdown("""
    Dataset berasal dari UCI Machine Learning Repository:
    https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
    """)

    st.subheader("📌 Range Normal Fitur")

    st.dataframe(
        pd.DataFrame.from_dict(
            normal_range,
            orient="index",
            columns=["Nilai Normal"]
        ),
        use_container_width=True
    )

# ==========================================
# PREDIKSI
# ==========================================
elif menu == "Prediksi":

    st.title("🔍 Prediksi Gangguan Hati")

    st.info("Input boleh bernilai 0 atau lebih (>= 0)")

    def safe_input(label):
        return st.number_input(label, min_value=0.0, value=0.0)

    col1, col2 = st.columns(2)

    with col1:
        tb = safe_input("Total Bilirubin")
        db = safe_input("Direct Bilirubin")
        alt = safe_input("ALT")

    with col2:
        ast = safe_input("AST")
        tp = safe_input("Total Proteins")
        alb = safe_input("Albumin")

    input_df = pd.DataFrame([[tb, db, alt, ast, tp, alb]], columns=[
        'Total_Bilirubin',
        'Direct_Bilirubin',
        'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase',
        'Total_Proteins',
        'Albumin'
    ])

    st.subheader("Data Input")
    st.dataframe(input_df)

    if st.button("🔬 Prediksi"):

        pred = model.predict(input_df)[0]

        # Jika model mendukung probabilitas
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0]

            st.subheader("Hasil Prediksi")

            if pred == 1:
                st.error("🔴 Terindikasi Gangguan Hati")
            else:
                st.success("🟢 Normal")

            st.write(f"Probabilitas Normal: {prob[0]*100:.2f}%")
            st.write(f"Probabilitas Gangguan: {prob[1]*100:.2f}%")

        else:
            st.subheader("Hasil Prediksi")

            if pred == 1:
                st.error("🔴 Terindikasi Gangguan Hati")
            else:
                st.success("🟢 Normal")

# ==========================================
# HEALTH INFO
# ==========================================
elif menu == "Kesehatan Hati":

    st.title("🫀 Edukasi Kesehatan Hati")

    st.markdown("""
    ### 📌 Fungsi hati
    - Detoksifikasi racun
    - Metabolisme nutrisi
    - Produksi empedu
    """)

    st.markdown("""
    ### ⚠️ Penyebab gangguan hati
    - Alkohol
    - Hepatitis
    - Obesitas
    - Pola makan buruk
    """)

    st.success("Menjaga hati = menjaga kesehatan tubuh")

    df_info = pd.DataFrame({
        "Fitur": list(normal_range.keys()),
        "Nilai Normal": list(normal_range.values()),
        "Fungsi": [
            "Sisa pemecahan darah",
            "Bilirubin terkonjugasi",
            "Enzim kerusakan hati",
            "Enzim metabolisme",
            "Protein darah",
            "Protein hati"
        ],
        "Interpretasi": [
            "Naik → gangguan hati",
            "Naik → sumbatan empedu",
            "Naik → kerusakan sel hati",
            "Naik → gangguan hati/otot",
            "Rendah → gangguan hati",
            "Rendah → fungsi hati menurun"
        ]
    })

    st.dataframe(df_info, use_container_width=True)
