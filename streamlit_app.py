import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="ILPD Liver Disease App",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# NAVIGATION
# ==========================================
menu = st.sidebar.selectbox(
    "📌 Menu",
    ["Home", "Dataset ILPD", "Prediksi", "Kesehatan Hati"]
)

# ==========================================
# TRAIN MODEL
# ==========================================
@st.cache_resource
def train_model():

    df = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv", header=None)

    df.columns = [
        'Age','Gender','Total_Bilirubin','Direct_Bilirubin',
        'Alkaline_Phosphotase','Alamine_Aminotransferase',
        'Aspartate_Aminotransferase','Total_Proteins',
        'Albumin','Albumin_and_Globulin_Ratio','Target'
    ]

    df = df.drop_duplicates()

    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Target'] = df['Target'].map({1: 1, 2: 0})

    df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(
        df['Albumin_and_Globulin_Ratio'].mean()
    )

    features = [
        'Total_Bilirubin','Direct_Bilirubin',
        'Alamine_Aminotransferase','Aspartate_Aminotransferase',
        'Total_Proteins','Albumin'
    ]

    X_raw = df[features]
    y = df['Target']

    scaler = QuantileTransformer()
    X = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier()

    param_grid = {
        "n_neighbors": list(range(3, 25)),
        "p": [1, 2],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)

    grid = GridSearchCV(
        knn,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, scaler


model, scaler = train_model()

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

    st.info("Machine Learning KNN untuk klasifikasi penyakit hati")

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
    Dataset ini berasal dari UCI Machine Learning Repository.

    🔗 Link:
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

    st.warning("Semua input wajib > 0")

    def safe_input(label, min_v, max_v, default):
        val = st.number_input(label, min_value=min_v, max_value=max_v, value=default)
        if val == 0:
            st.error(f"{label} tidak boleh 0")
        return val

    col1, col2 = st.columns(2)

    with col1:
        tb = safe_input("Total Bilirubin", 0.1, 75.0, 1.0)
        db = safe_input("Direct Bilirubin", 0.1, 20.0, 0.3)
        alt = safe_input("ALT", 1, 2000, 30)

    with col2:
        ast = safe_input("AST", 1, 5000, 30)
        tp = safe_input("Total Proteins", 2.0, 10.0, 7.0)
        alb = safe_input("Albumin", 0.5, 6.5, 4.0)

    input_df = pd.DataFrame([[
        tb, db, alt, ast, tp, alb
    ]], columns=[
        'Total_Bilirubin','Direct_Bilirubin',
        'Alamine_Aminotransferase','Aspartate_Aminotransferase',
        'Total_Proteins','Albumin'
    ])

    st.subheader("Data Input")
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

    st.subheader("📊 Informasi Fitur ILPD")

    df_info = pd.DataFrame({
        "Fitur": [
            "Total Bilirubin",
            "Direct Bilirubin",
            "ALT",
            "AST",
            "Total Proteins",
            "Albumin"
        ],
        "Nilai Normal": [
            "0.1 - 1.2 mg/dL",
            "0.0 - 0.3 mg/dL",
            "7 - 56 U/L",
            "10 - 40 U/L",
            "6.0 - 8.3 g/dL",
            "3.5 - 5.0 g/dL"
        ],
        "Fungsi": [
            "Sisa pemecahan darah yang diproses hati",
            "Bilirubin siap dibuang oleh hati",
            "Enzim indikator kerusakan hati",
            "Enzim metabolisme organ tubuh",
            "Protein untuk imun & tekanan darah",
            "Protein utama dari hati"
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
