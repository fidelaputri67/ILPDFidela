# -*- coding: utf-8 -*-
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
# LOAD MODEL + SCALER
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load("ilpd_best_model.pkl")
    scaler = joblib.load("quantile_transformer.pkl")
    return model, scaler

model, scaler = load_model()

# ==========================================
# MENU
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
    st.title("🩺 ILPD Liver Disease Prediction")

    st.info("Model Machine Learning (Naive Bayes) untuk prediksi penyakit hati")

    st.markdown("""
    ### Fitur:
    - Prediksi penyakit hati
    - Informasi dataset
    - Edukasi kesehatan hati
    """)

# ==========================================
# DATASET
# ==========================================
elif menu == "Dataset ILPD":

    st.title("📊 Dataset ILPD")

    st.markdown("Dataset dari UCI Machine Learning Repository")

    st.subheader("Nilai Normal Fitur")

    st.dataframe(
        pd.DataFrame.from_dict(
            normal_range,
            orient="index",
            columns=["Range Normal"]
        ),
        use_container_width=True
    )

# ==========================================
# PREDIKSI
# ==========================================
elif menu == "Prediksi":

    st.title("🔍 Prediksi Penyakit Hati")

    st.warning("Input boleh 0, tapi pastikan sesuai kondisi medis")

    def input_num(label):
        return st.number_input(label, min_value=0.0, value=0.0, step=0.1)

    col1, col2 = st.columns(2)

    with col1:
        tb = input_num("Total Bilirubin")
        db = input_num("Direct Bilirubin")
        alt = input_num("ALT (Alamine Aminotransferase)")

    with col2:
        ast = input_num("AST (Aspartate Aminotransferase)")
        tp = input_num("Total Proteins")
        alb = input_num("Albumin")

    input_df = pd.DataFrame([[tb, db, alt, ast, tp, alb]],
        columns=[
            'Total_Bilirubin',
            'Direct_Bilirubin',
            'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase',
            'Total_Proteins',
            'Albumin'
        ]
    )

    st.subheader("Input Data")
    st.dataframe(input_df)

    if st.button("🔬 Prediksi"):

        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0]

        st.subheader("Hasil Prediksi")

        if pred == 1:
            st.error("🔴 Terindikasi Gangguan Hati")
        else:
            st.success("🟢 Normal")

        st.write(f"Probabilitas Normal: {prob[0]*100:.2f}%")
        st.write(f"Probabilitas Gangguan: {prob[1]*100:.2f}%")

# ==========================================
# HEALTH INFO
# ==========================================
elif menu == "Kesehatan Hati":

    st.title("🫀 Edukasi Kesehatan Hati")

    st.markdown("""
    ### Fungsi hati:
    - Detoksifikasi racun
    - Metabolisme nutrisi
    - Produksi empedu
    """)

    st.markdown("""
    ### Penyebab gangguan hati:
    - Alkohol
    - Hepatitis
    - Obesitas
    - Pola makan buruk
    """)

    st.success("Menjaga hati = menjaga kesehatan tubuh")

    df_info = pd.DataFrame({
        "Fitur": [
            "Bilirubin", "Direct Bilirubin", "ALT",
            "AST", "Total Proteins", "Albumin"
        ],
        "Fungsi": [
            "Sisa pemecahan darah",
            "Bentuk bilirubin siap dibuang",
            "Enzim kerusakan hati",
            "Enzim metabolisme tubuh",
            "Protein imun tubuh",
            "Protein utama hati"
        ]
    })

    st.dataframe(df_info, use_container_width=True)
