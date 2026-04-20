"""
Heart Disease Predictor — Capstone Project
Bootcamp Machine Learning & AI for Beginner | DQLab × Live
Author : Ahmad Ihsan Fuady
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8F9FF; }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #3D35A8 0%, #5549CC 100%);
        border-radius: 16px;
        padding: 32px 40px;
        color: white;
        margin-bottom: 28px;
    }
    .header-card h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .header-card p  { font-size: 1rem; opacity: 0.85; margin: 8px 0 0; }
    .badge {
        background: #B5E31C;
        color: #1E1A5E;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 12px;
    }

    /* Prediction result cards */
    .result-positive {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #22C55E, #16A34A);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        text-align: center;
    }
    .result-card h2 { font-size: 1.8rem; margin: 8px 0 4px; }
    .result-card p  { opacity: 0.9; font-size: 0.95rem; margin: 0; }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 12px;
        margin-top: 16px;
    }
    .metric-card {
        background: #3D35A8;
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        flex: 1;
        text-align: center;
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 800; color: #B5E31C; }
    .metric-card .lbl { font-size: 0.75rem; opacity: 0.8; margin-top: 4px; }

    /* Info box */
    .info-box {
        background: #EEF2FF;
        border-left: 4px solid #3D35A8;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.9rem;
        color: #374151;
        margin-top: 12px;
    }

    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #3D35A8;
        margin: 20px 0 8px;
        border-bottom: 2px solid #B5E31C;
        padding-bottom: 4px;
    }

    /* Hide Streamlit default header */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Load & Train Model (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Melatih model... harap tunggu sebentar.")
def load_model():
    url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
    df  = pd.read_csv(url).dropna()

    X_raw = df.drop("target", axis=1)
    y     = df["target"].values

    # PCA (n=9) → StandardScaler → Decision Tree
    pca    = PCA(n_components=9)
    X_pca  = pca.fit_transform(X_raw)

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_pca)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=42)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)

    return model, pca, scaler, df, accuracy, report, X_raw.columns.tolist()


model, pca, scaler, df, accuracy, report, feature_cols = load_model()


# ─────────────────────────────────────────────────────────────────────
# Predict function
# ─────────────────────────────────────────────────────────────────────
def predict(input_dict: dict):
    input_df  = pd.DataFrame([input_dict])
    input_pca = pca.transform(input_df)
    input_sc  = scaler.transform(input_pca)
    pred      = model.predict(input_sc)[0]
    proba     = model.predict_proba(input_sc)[0]
    return pred, proba


# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="badge">🎓 Capstone Project — DQLab × Live</div>
    <h1>🫀 Heart Disease Predictor</h1>
    <p>Prediksi risiko penyakit jantung menggunakan Machine Learning (Decision Tree + PCA)<br>
    <strong>Ahmad Ihsan Fuady</strong> · Bootcamp Machine Learning &amp; AI for Beginner</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar — Input Features
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Input Fitur Pasien")
    st.markdown("Isi data klinis pasien di bawah ini:")

    st.markdown('<div class="section-title">👤 Demografi</div>', unsafe_allow_html=True)
    age = st.slider("Usia (tahun)", 20, 80, 54)
    sex = st.selectbox("Jenis Kelamin", options=[0, 1],
                       format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")

    st.markdown('<div class="section-title">💊 Kondisi Klinis</div>', unsafe_allow_html=True)
    cp = st.selectbox("Tipe Nyeri Dada (cp)",
                      options=[0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "0 — Typical Angina",
                          1: "1 — Atypical Angina",
                          2: "2 — Non-anginal Pain",
                          3: "3 — Asymptomatic",
                      }[x])
    trestbps = st.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 130)
    chol     = st.slider("Kolesterol Serum (mg/dl)", 100, 600, 246)
    fbs      = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)",
                             options=[0, 1],
                             format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
    restecg  = st.selectbox("Hasil EKG Istirahat (restecg)",
                             options=[0, 1, 2],
                             format_func=lambda x: {
                                 0: "0 — Normal",
                                 1: "1 — ST-T Abnormality",
                                 2: "2 — LV Hypertrophy",
                             }[x])

    st.markdown('<div class="section-title">🏃 Hasil Uji Stres</div>', unsafe_allow_html=True)
    thalach = st.slider("Detak Jantung Maks. (bpm)", 60, 220, 150)
    exang   = st.selectbox("Angina saat Olahraga (exang)",
                            options=[0, 1],
                            format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope   = st.selectbox("Slope ST Segment",
                            options=[0, 1, 2],
                            format_func=lambda x: {
                                0: "0 — Upsloping",
                                1: "1 — Flat",
                                2: "2 — Downsloping",
                            }[x])

    st.markdown('<div class="section-title">🔬 Angiografi</div>', unsafe_allow_html=True)
    ca   = st.selectbox("Jumlah Pembuluh Darah Mayor (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Hasil Thalassemia (thal)",
                         options=[0, 1, 2, 3],
                         format_func=lambda x: {
                             0: "0 — Normal",
                             1: "1 — Fixed Defect",
                             2: "2 — Normal (2)",
                             3: "3 — Reversable Defect",
                         }[x])

    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────
# Main — Tabs
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi", "📊 Data & EDA", "🤖 Performa Model"])

# ── TAB 1: PREDICTION ────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown("### 📋 Ringkasan Data Pasien")
        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg,
            "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
            "slope": slope, "ca": ca, "thal": thal,
        }
        summary_df = pd.DataFrame({
            "Fitur": ["Usia", "Jenis Kelamin", "Tipe Nyeri Dada", "TD Istirahat",
                      "Kolesterol", "Gula Darah Puasa", "EKG Istirahat",
                      "Detak Jantung Maks", "Angina Olahraga", "ST Depression",
                      "Slope ST", "Pembuluh Mayor", "Thalassemia"],
            "Nilai": [age,
                      "Laki-laki" if sex == 1 else "Perempuan",
                      f"Tipe {cp}", f"{trestbps} mmHg",
                      f"{chol} mg/dl",
                      "Ya" if fbs == 1 else "Tidak",
                      f"Kode {restecg}",
                      f"{thalach} bpm", "Ya" if exang == 1 else "Tidak",
                      f"{oldpeak}", f"Slope {slope}", ca, thal],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("### 🎯 Hasil Prediksi")

        if predict_btn or True:   # Show default state on first load
            pred, proba = predict(input_data)

            if pred == 1:
                st.markdown("""
                <div class="result-positive result-card">
                    <div style="font-size:3rem">⚠️</div>
                    <h2>Terindikasi Penyakit Jantung</h2>
                    <p>Model memprediksi pasien berisiko tinggi.<br>
                    Segera konsultasikan ke dokter spesialis.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative result-card">
                    <div style="font-size:3rem">✅</div>
                    <h2>Tidak Terindikasi</h2>
                    <p>Model memprediksi pasien tidak berisiko tinggi.<br>
                    Tetap jaga pola hidup sehat!</p>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            st.markdown("#### 📈 Probabilitas Prediksi")
            prob_df = pd.DataFrame({
                "Kondisi": ["Tidak Sakit Jantung", "Sakit Jantung"],
                "Probabilitas": [round(proba[0] * 100, 1), round(proba[1] * 100, 1)],
            })
            st.bar_chart(prob_df.set_index("Kondisi"), color=["#3D35A8"])

            st.markdown(f"""
            <div class="info-box">
            ⚠️ <strong>Disclaimer:</strong> Prediksi ini dihasilkan oleh model Machine Learning
            dan <strong>bukan merupakan diagnosis medis</strong>. Selalu konsultasikan hasil ini
            dengan tenaga medis profesional.
            </div>
            """, unsafe_allow_html=True)


# ── TAB 2: DATA & EDA ────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Eksplorasi Data Heart Disease")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(df):,} rows")
    col2.metric("Total Fitur", f"{len(feature_cols)} fitur")
    col3.metric("Kasus Positif", f"{df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    col4.metric("Kasus Negatif", f"{(df['target']==0).sum()} ({(1-df['target'].mean())*100:.1f}%)")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📋 Statistik Deskriptif")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with col_b:
        st.markdown("#### 🔢 5 Baris Teratas Dataset")
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("#### 🔥 Distribusi Target")
    target_counts = df["target"].value_counts().reset_index()
    target_counts.columns = ["Target", "Jumlah"]
    target_counts["Target"] = target_counts["Target"].map({0: "Tidak Sakit (0)", 1: "Sakit Jantung (1)"})
    st.bar_chart(target_counts.set_index("Target"), color=["#3D35A8"])

    st.markdown("#### 📊 Pilih Fitur untuk Distribusi")
    selected_feat = st.selectbox("Fitur:", feature_cols)
    st.bar_chart(df[selected_feat].value_counts().sort_index())


# ── TAB 3: MODEL PERFORMANCE ─────────────────────────────────────────
with tab3:
    st.markdown("### 🤖 Performa Model — Decision Tree Classifier")

    st.markdown("""
    <div class="info-box">
    Pipeline: <strong>Raw Data (13 fitur)</strong> → <strong>PCA (9 komponen)</strong>
    → <strong>StandardScaler</strong> → <strong>DecisionTreeClassifier (gini, max_depth=30)</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Accuracy (Test)", f"{accuracy*100:.2f}%")
    c2.metric("🔍 PCA Components", "9 / 13")
    c3.metric("🌳 Model", "Decision Tree")

    st.markdown("#### 📄 Classification Report (Test Set)")
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("#### ⚙️ Parameter Model")
    params_df = pd.DataFrame({
        "Parameter": ["criterion", "max_depth", "random_state", "test_size", "PCA n_components"],
        "Nilai":     ["gini",      "30",         "42",            "0.2 (80:20)", "9"],
    })
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📌 Fitur yang Digunakan (13 Fitur Asli)")
    feat_df = pd.DataFrame({
        "Fitur": feature_cols,
        "Deskripsi": [
            "Usia pasien (tahun)",
            "Jenis kelamin (0=Perempuan, 1=Laki-laki)",
            "Tipe nyeri dada (0-3)",
            "Tekanan darah istirahat (mmHg)",
            "Kolesterol serum (mg/dl)",
            "Gula darah puasa > 120 mg/dl",
            "Hasil EKG istirahat (0-2)",
            "Detak jantung maksimum",
            "Angina akibat olahraga",
            "ST depression (oldpeak)",
            "Slope segmen ST (0-2)",
            "Jumlah pembuluh mayor (0-3)",
            "Hasil thalassemia (0-3)",
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748B; font-size:0.85rem; padding: 8px 0 16px;">
    🫀 <strong>Heart Disease Predictor</strong> · Capstone Project ML & AI<br>
    Ahmad Ihsan Fuady · Bootcamp DQLab × Live · 2025
</div>
""", unsafe_allow_html=True)
