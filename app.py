import streamlit as st
import pickle
import pandas as pd
import time

st.set_page_config(page_title="ML Portfolio | Ahmad Ihsan Fuady", page_icon="🤖", layout="wide")

select_var = st.sidebar.selectbox("Select page", ["Home", "Iris Species", "Heart Disease"])

# ===================== HOME =====================
if select_var == "Home":
    st.title("🤖 ML Portfolio — Ahmad Ihsan Fuady")
    st.markdown("**Information Systems Graduate · Data Science Enthusiast · Telkomsel**")

    st.markdown("---")

    st.subheader("👋 About Me")
    st.write("""
    Portofolio ini mendemonstrasikan implementasi Machine Learning end-to-end:
    mulai dari preprocessing, modeling, evaluasi hingga deployment menggunakan Streamlit.
    """)

    st.subheader("🚀 Models")
    st.info("🌸 Iris Classifier — SVM + GridSearchCV")
    st.error("❤️ Heart Disease Predictor — MLP + Threshold Optimization")


# ===================== IRIS =====================
elif select_var == "Iris Species":
    st.title("🌸 Iris Species Prediction")

    st.sidebar.header('Input Features')

    SepalLengthCm = st.sidebar.slider('Sepal Length', 4.3, 7.9, 6.5)
    SepalWidthCm = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.3)
    PetalLengthCm = st.sidebar.slider('Petal Length', 1.0, 6.9, 4.5)
    PetalWidthCm = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.4)

    df = pd.DataFrame({
        'SepalLengthCm': [SepalLengthCm],
        'SepalWidthCm': [SepalWidthCm],
        'PetalLengthCm': [PetalLengthCm],
        'PetalWidthCm': [PetalWidthCm]
    })

    st.write("**Input Data:**")
    st.write(df)

    if st.sidebar.button("Predict"):
        with open("generate_iris.pkl", "rb") as f:
            model = pickle.load(f)

        pred = model.predict(df.values)[0]
        species = ["Setosa", "Versicolor", "Virginica"][pred]

        st.success(f"🌸 Prediction: {species}")

        # Interpretasi sederhana
        st.subheader("📖 Interpretasi")
        if df['PetalLengthCm'][0] < 2:
            st.write("- Petal kecil → kemungkinan Setosa")
        elif df['PetalLengthCm'][0] < 5:
            st.write("- Petal sedang → kemungkinan Versicolor")
        else:
            st.write("- Petal besar → kemungkinan Virginica")


# ===================== HEART DISEASE =====================
elif select_var == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")

    st.sidebar.header('Input Features')

    cp = st.sidebar.selectbox("Chest Pain", [0,1,2,3])
    thalach = st.sidebar.slider("Max Heart Rate", 71, 202, 150)
    slope = st.sidebar.selectbox("Slope", [0,1,2])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.2, 1.0)
    exang = st.sidebar.selectbox("Exercise Angina", [0,1])
    ca = st.sidebar.selectbox("Major Vessels", [0,1,2,3])
    thal = st.sidebar.selectbox("Thal", [1,2,3])
    sex = st.sidebar.selectbox("Sex", [0,1])
    age = st.sidebar.slider("Age", 29, 77, 50)

    df = pd.DataFrame({
        'cp': [cp],
        'thalach': [thalach],
        'slope': [slope],
        'oldpeak': [oldpeak],
        'exang': [exang],
        'ca': [ca],
        'thal': [thal],
        'sex': [sex],
        'age': [age]
    })

    st.write("**Input Data:**")
    st.write(df)

    if st.sidebar.button("Predict"):
        with open("full_heart_disease_pipeline.pkl", "rb") as f:
            model = pickle.load(f)

        prob = model.predict_proba(df.values)[0][1]
        pred = 1 if prob >= 0.4 else 0

        st.subheader("Prediction Result")

        if pred == 1:
            st.error("⚠️ Heart Disease Risk Detected")
        else:
            st.success("✅ No Risk")

        st.metric("Probability", f"{prob*100:.1f}%")

        # ================= INTERPRETASI PER ROW =================
        st.subheader("📖 Interpretasi Input")

        explanations = []

        if cp == 3:
            explanations.append("Chest pain asymptomatic → sering terkait risiko tinggi")

        if thalach < 120:
            explanations.append("Detak jantung maksimum rendah")

        if oldpeak > 2:
            explanations.append("ST depression tinggi → indikasi ischemia")

        if exang == 1:
            explanations.append("Nyeri saat exercise")

        if ca >= 2:
            explanations.append("Banyak pembuluh darah terdeteksi")

        if thal == 3:
            explanations.append("Reversible defect")

        if age > 55:
            explanations.append("Usia tinggi → faktor risiko")

        if explanations:
            for e in explanations:
                st.write(f"- {e}")
        else:
            st.write("- Tidak ada indikator risiko signifikan")

        # ================= MODEL INSIGHT =================
        st.subheader("🧠 Model Insight")

        risk_factors = []

        if oldpeak > 2:
            risk_factors.append("ST depression tinggi")

        if exang == 1:
            risk_factors.append("Exercise angina")

        if ca >= 2:
            risk_factors.append("Multiple vessels")

        if thal == 3:
            risk_factors.append("Reversible defect")

        if pred == 1:
            if risk_factors:
                st.warning("Prediksi dipengaruhi oleh:")
                for r in risk_factors:
                    st.write(f"- {r}")
            else:
                st.warning("Dipengaruhi kombinasi fitur")
        else:
            st.success("Tidak ada faktor signifikan")

        st.caption("⚠️ Model ini hanya decision support, bukan diagnosis medis.")
