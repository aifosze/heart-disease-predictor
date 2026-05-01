import streamlit as st
import pickle
import pandas as pd 
import time

st.set_page_config(page_title="ML Portfolio | Ahmad Ihsan Fuady", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

select_var = st.sidebar.selectbox("Select page", ["Home", "Iris Species", "Heart Disease"])

if select_var == "Home":
    st.title("🤖 ML Portfolio — Ahmad Ihsan Fuady")
    st.markdown("**Information Systems Graduate · Data Science Enthusiast · Telkomsel**")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("👋 About Me")
        st.write("""
        Hi! Saya **Ahmad Ihsan Fuady**, lulusan Sistem Informasi Universitas Telkom Surabaya 
        dengan IPK 3.78 (Cumlaude). Saat ini bekerja di divisi **Mobile Broadband Assurance** 
        Telkomsel sambil aktif mengembangkan kompetensi di bidang **Data Science & Machine Learning**.

        Portofolio ini mendemonstrasikan implementasi ML untuk klasifikasi data nyata — 
        mulai dari dataset klasik Iris hingga prediksi risiko penyakit jantung.
        """)

        st.subheader("🏆 Highlights")
        st.write("""
        - 📄 Publikasi **Scopus-indexed** IEEE ICITISEE 2022 (Lean-UX Mobile App Research)
        - 📄 Dua publikasi jurnal Sinta 3 (SISTEMASI 2023, JIPI 2025 — K-Means clustering)
        - 🎓 Bangkit Academy ML Path — K-Means + Random Forest capstone
        - 💼 Data Science Virtual Internship — ID/X Partners via Rakamin (AUC-ROC 0.71)
        - 👨‍🏫 Teaching Assistant Big Data Analytics — Telkom University Surabaya
        """)

        st.subheader("🛠️ Tech Stack")
        st.write("`Python` `Scikit-learn` `Pandas` `Streamlit` `TensorFlow` `Gephi` `SQL`")

        st.markdown("---")
        st.subheader("🔗 Connect with Me")
        col_gh, col_li = st.columns(2)
        with col_gh:
            st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-aifosze-181717?style=for-the-badge&logo=github)](https://github.com/aifosze)")
        with col_li:
            st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ahmad%20Ihsan%20Fuady-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/ahmadihsanfuady)")

    with col2:
        st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png", width=280)
        st.markdown("---")
        st.subheader("📊 Models in This App")
        st.info("🌸 **Iris Species Classifier**\nSVM + StandardScaler Pipeline\nGridSearchCV Hyperparameter Tuning")
        st.error("❤️ **Heart Disease Predictor**\nMLP Classifier + StandardScaler\n9 fitur klinis, threshold 0.4")

    st.markdown("---")
    st.subheader("🚀 Getting Started")
    st.write("""
    1. Pilih model dari menu **sidebar** di sebelah kiri
    2. Input data:
       - **Upload CSV** — batch prediction dengan dataset Anda
       - **Manual Input** — gunakan slider/input untuk satu prediksi
    3. Klik tombol **Submit / Predict!**
    4. Hasil prediksi muncul secara instant ✅
    """)
    st.info("""
    💡 **Format CSV:**
    - Iris: kolom `SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm`
    - Heart Disease: kolom `cp, thalach, slope, oldpeak, exang, ca, thal, sex, age`
    """)

elif select_var == "Iris Species":
    st.title("🌸 Iris Species Prediction")
    st.write("""
    Prediksi jenis bunga Iris berdasarkan 4 pengukuran morfologi menggunakan **SVM Classifier** 
    dengan **GridSearchCV Hyperparameter Tuning** (Sesi 11 DQLab).

    Dataset: [UCI Iris Dataset via Kaggle](https://www.kaggle.com/uciml/iris)
    """)

    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', min_value=4.3, value=6.5, max_value=10.0)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, value=3.3, max_value=5.0)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)', min_value=1.0, value=4.5, max_value=9.0)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)', min_value=0.1, value=1.4, max_value=5.0)
            data = {'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

    button_var = st.sidebar.button('Submit')

    st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png", width=500)

    if button_var:
        df = input_df
        # Ambil hanya 4 kolom yang dibutuhkan (handle CSV dari Kaggle yang ada kolom Id)
        feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        df = df[feature_cols]
        st.write("**Input Data:**")
        st.write(df)

        with open("generate_iris.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
            prediction = loaded_model.predict(df.values)
            result = 'Iris-setosa' if prediction[0] == 0 else ('Iris-versicolor' if prediction[0] == 1 else 'Iris-virginica')
            st.subheader('Prediction Result:')
            with st.spinner('Analyzing...'):
                time.sleep(2)
                st.success(f"🌸 Predicted Species: **{result}**")

elif select_var == "Heart Disease":
    st.title("❤️ Heart Disease Risk Prediction")
    st.write("""
    Prediksi risiko penyakit jantung berdasarkan **9 fitur klinis** menggunakan **MLP Classifier**
    dengan **GridSearchCV Hyperparameter Tuning** dan analisis ROC (Sesi 13–14 DQLab).

    Dataset: [Heart Disease UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) — Threshold: **0.4**
    """)

    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')

            # ✅ Urutan input sesuai Sesi 13: cp, thalach, slope, oldpeak, exang, ca, thal, sex, age
            chest_pain_map = {
                "Typical Angina (0)": 0,
                "Atypical Angina (1)": 1,
                "Non-Anginal Pain (2)": 2,
                "Asymptomatic (3)": 3
            }
            wcp = st.sidebar.selectbox('Chest Pain Type', options=list(chest_pain_map.keys()), help="Jenis nyeri dada yang dirasakan pasien")
            cp = chest_pain_map[wcp]

            thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=71, value=150, max_value=202, help="Detak jantung maksimum saat exercise")
            slope = st.sidebar.selectbox('Slope of ST Segment', options=[0, 1, 2], index=1, help="Kemiringan segmen ST pada EKG")
            oldpeak = st.sidebar.slider('Oldpeak (ST Depression)', min_value=0.0, value=1.0, max_value=6.2, step=0.1, help="Seberapa banyak ST segmen menurun")
            exang = st.sidebar.radio('Exercise Induced Angina', options=['Yes (1)', 'No (0)'], index=1, help="Apakah terjadi angina saat exercise?")
            exang = 1 if exang == 'Yes (1)' else 0
            ca = st.sidebar.selectbox('Number of Major Vessels', options=[0, 1, 2, 3], index=0, help="Jumlah pembuluh darah utama (fluoroskopi)")
            thal = st.sidebar.selectbox('Thalassemia', options=[1, 2, 3], index=0, help="Hasil tes thalium: 1=normal, 2=fixed defect, 3=reversible defect")
            sex = st.sidebar.radio('Sex', options=['Male (1)', 'Female (0)'], index=0)
            sex = 1 if sex == 'Male (1)' else 0
            age = st.sidebar.slider('Age', min_value=29, value=50, max_value=77, step=1, help="Usia pasien")

            # ✅ Urutan dict sesuai training Sesi 13
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}

            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

    st.image("https://drramjimehrotra.com/wp-content/uploads/2022/09/Women-Heart-Disease-min-resize.png", width=300)

    if st.sidebar.button('Predict!'):
        df = input_df

        # ✅ Urutan feature_cols sesuai Sesi 13
        feature_cols = ['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age']
        df = df[feature_cols]

        st.write("**Input Data:**")
        st.write(df)

        with open("full_heart_disease_pipeline.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        prediction_proba = loaded_model.predict_proba(df.values.astype(float))
        prob_risk = prediction_proba[:, 1][0]

        # Threshold 0.4 sesuai analisis ROC Sesi 14
        prediction = 1 if prob_risk >= 0.4 else 0
        result = 'No Heart Disease Risk' if prediction == 0 else 'Heart Disease Risk Detected'

        st.subheader('Prediction Result:')
        with st.spinner('Analyzing...'):
            time.sleep(2)
            if result == "No Heart Disease Risk":
                st.success(f"✅ **{result}**")
            else:
                st.error(f"⚠️ **{result}**")
                st.info("Harap konsultasikan hasil ini dengan dokter untuk evaluasi lebih lanjut.")
            st.metric(
                label="Probability of Heart Disease Risk",
                value="{:.1f}%".format(prob_risk * 100),
                delta="Above threshold (40%)" if prob_risk >= 0.4 else "Below threshold (40%)",
                delta_color="inverse"
            )

# Footer sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Ahmad Ihsan Fuady**")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-aifosze-181717?logo=github)](https://github.com/aifosze) [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://linkedin.com/in/ahmadihsanfuady)")
