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
# LOAD MODEL PIPELINE (WAJIB)
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load("ilpd_best_model.pkl")
    return model

model = load_model()

# ==========================================
# MENU
# ==========================================
menu = st.sidebar.selectbox(
    "📌 Menu",
    ["Home", "Dataset ILPD", "Prediksi", "Kesehatan Hati"]
)

# ==========================================
# HOME
# ==========================================
if menu == "Home":
    st.title("🩺 Sistem Prediksi Penyakit Hati (ILPD)")
    st.info("Model ML berbasis pipeline (Scaler + Naive Bayes / KNN / best model)")

# ==========================================
# DATASET INFO
# ==========================================
elif menu == "Dataset ILPD":
    st.title("📊 Dataset ILPD")

    st.markdown("""
    Dataset UCI ILPD
    """)

# ==========================================
# PREDIKSI (FIX UTAMA)
# ==========================================
elif menu == "Prediksi":

    st.title("🔍 Prediksi Gangguan Hati")

    st.info("Input boleh 0 atau lebih (>= 0)")

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
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Proteins",
        "Albumin"
    ])

    st.write("Input Data")
    st.dataframe(input_df)

    if st.button("🔬 Prediksi"):

        # 🔥 PENTING: langsung predict tanpa manual scaler
        pred = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0]

            if pred == 1:
                st.error("🔴 Terindikasi Gangguan Hati")
            else:
                st.success("🟢 Normal")

            st.write(f"Prob Normal: {prob[0]*100:.2f}%")
            st.write(f"Prob Gangguan: {prob[1]*100:.2f}%")
        else:
            if pred == 1:
                st.error("🔴 Terindikasi Gangguan Hati")
            else:
                st.success("🟢 Normal")

# ==========================================
# EDUKASI
# ==========================================
elif menu == "Kesehatan Hati":
    st.title("🫀 Edukasi Kesehatan Hati")

    st.markdown("""
    - Detoksifikasi
    - Metabolisme
    - Produksi empedu
    """)
