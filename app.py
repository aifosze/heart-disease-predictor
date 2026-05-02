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

    Dataset: [UCI Iris Dataset via Kaggle](https://www.kaggle.com/uciml/iris) — 150 sampel, 3 kelas, 4 fitur numerik
    """)

    # Dataset feature info
    with st.expander("ℹ️ Tentang Fitur Dataset Iris (UCI)"):
        st.markdown("""
        | Fitur | Deskripsi | Range |
        |-------|-----------|-------|
        | **SepalLengthCm** | Panjang sepal (kelopak luar pelindung bunga) dalam cm | 4.3 – 7.9 cm |
        | **SepalWidthCm** | Lebar sepal dalam cm | 2.0 – 4.4 cm |
        | **PetalLengthCm** | Panjang petal (kelopak bunga berwarna bagian dalam) dalam cm | 1.0 – 6.9 cm |
        | **PetalWidthCm** | Lebar petal dalam cm | 0.1 – 2.5 cm |
        
        **Target kelas:** Iris-setosa (0) · Iris-versicolor (1) · Iris-virginica (2)
        
        💡 *Petal features terbukti lebih diskriminatif daripada Sepal features dalam membedakan ketiga spesies.*
        """)

    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider(
                'Sepal Length (cm)',
                min_value=4.3, value=6.5, max_value=7.9,
                help="Panjang kelopak luar (sepal) bunga Iris dalam sentimeter. Range: 4.3–7.9 cm"
            )
            SepalWidthCm = st.sidebar.slider(
                'Sepal Width (cm)',
                min_value=2.0, value=3.3, max_value=4.4,
                help="Lebar kelopak luar (sepal) bunga Iris dalam sentimeter. Range: 2.0–4.4 cm"
            )
            PetalLengthCm = st.sidebar.slider(
                'Petal Length (cm)',
                min_value=1.0, value=4.5, max_value=6.9,
                help="Panjang kelopak dalam berwarna (petal) bunga Iris dalam sentimeter. Range: 1.0–6.9 cm"
            )
            PetalWidthCm = st.sidebar.slider(
                'Petal Width (cm)',
                min_value=0.1, value=1.4, max_value=2.5,
                help="Lebar kelopak dalam berwarna (petal) bunga Iris dalam sentimeter. Range: 0.1–2.5 cm"
            )
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
    dengan **GridSearchCV Hyperparameter Tuning** dan analisis ROC threshold (Sesi 13–14 DQLab).

    Dataset: [Heart Disease UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) — 303 sampel, threshold probabilitas: **0.4**
    """)

    # Dataset feature info — corrected per UCI documentation
    with st.expander("ℹ️ Tentang Fitur Dataset Heart Disease (UCI) — 9 Fitur Terpilih"):
        st.markdown("""
        | Fitur | Deskripsi | Nilai |
        |-------|-----------|-------|
        | **cp** | Chest Pain Type — jenis nyeri dada | 0=Typical Angina · 1=Atypical Angina · 2=Non-Anginal Pain · 3=Asymptomatic |
        | **thalach** | Maximum Heart Rate Achieved — detak jantung maksimum saat tes | 71 – 202 bpm |
        | **slope** | Slope of Peak Exercise ST Segment — kemiringan segmen ST di EKG | 0=Upsloping · 1=Flat · 2=Downsloping |
        | **oldpeak** | ST Depression — penurunan segmen ST akibat exercise vs istirahat | 0.0 – 6.2 |
        | **exang** | Exercise Induced Angina — nyeri dada saat olahraga | 0=Tidak · 1=Ya |
        | **ca** | Number of Major Vessels — jumlah pembuluh darah utama (fluoroskopi) | 0 · 1 · 2 · 3 |
        | **thal** | Thalassemia — hasil tes thalium jantung | 1=Normal · 2=Fixed Defect · 3=Reversible Defect |
        | **sex** | Jenis kelamin pasien | 0=Female · 1=Male |
        | **age** | Usia pasien dalam tahun | 29 – 77 tahun |
        
        💡 *Fitur terpilih berdasarkan analisis korelasi Sesi 13: cp, thalach, slope berkorelasi positif kuat; oldpeak, exang, ca, thal, sex, age berkorelasi cukup kuat dengan target.*
        """)

    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')

            # cp: 0–3 per UCI documentation (bukan 1–4)
            chest_pain_map = {
                "0 — Typical Angina": 0,
                "1 — Atypical Angina": 1,
                "2 — Non-Anginal Pain": 2,
                "3 — Asymptomatic": 3
            }
            wcp = st.sidebar.selectbox(
                'Chest Pain Type (cp)',
                options=list(chest_pain_map.keys()),
                help="Jenis nyeri dada. Typical Angina = nyeri terkait jantung saat aktivitas; Asymptomatic = tidak ada nyeri meski ada masalah jantung"
            )
            cp = chest_pain_map[wcp]

            thalach = st.sidebar.slider(
                'Max Heart Rate Achieved (thalach)',
                min_value=71, value=150, max_value=202,
                help="Detak jantung maksimum yang tercapai saat tes exercise (bpm). Sumber: UCI Heart Disease Dataset"
            )

            # slope: 0=Upsloping, 1=Flat, 2=Downsloping per UCI documentation
            slope_map = {
                "0 — Upsloping (ST naik)": 0,
                "1 — Flat (ST datar)": 1,
                "2 — Downsloping (ST turun)": 2
            }
            wslope = st.sidebar.selectbox(
                'Slope of ST Segment (slope)',
                options=list(slope_map.keys()),
                index=1,
                help="Kemiringan segmen ST pada EKG saat puncak exercise. Downsloping = indikator risiko lebih tinggi"
            )
            slope = slope_map[wslope]

            oldpeak = st.sidebar.slider(
                'ST Depression (oldpeak)',
                min_value=0.0, value=1.0, max_value=6.2, step=0.1,
                help="Penurunan segmen ST akibat exercise dibandingkan kondisi istirahat. Nilai tinggi = risiko lebih besar"
            )

            exang_map = {"0 — Tidak (No)": 0, "1 — Ya (Yes)": 1}
            wexang = st.sidebar.selectbox(
                'Exercise Induced Angina (exang)',
                options=list(exang_map.keys()),
                help="Apakah terjadi nyeri dada (angina) saat exercise? Ya = faktor risiko penyakit jantung"
            )
            exang = exang_map[wexang]

            ca = st.sidebar.selectbox(
                'Number of Major Vessels (ca)',
                options=[0, 1, 2, 3],
                help="Jumlah pembuluh darah utama yang terlihat via fluoroskopi (0–3). Makin banyak = kondisi jantung lebih buruk"
            )

            # thal: 1=Normal, 2=Fixed Defect, 3=Reversible Defect per UCI documentation
            thal_map = {
                "1 — Normal": 1,
                "2 — Fixed Defect": 2,
                "3 — Reversible Defect": 3
            }
            wthal = st.sidebar.selectbox(
                'Thalassemia (thal)',
                options=list(thal_map.keys()),
                help="Hasil tes thalium untuk memeriksa aliran darah jantung. Fixed Defect = kerusakan permanen; Reversible = muncul saat stress"
            )
            thal = thal_map[wthal]

            sex_map = {"0 — Female (Perempuan)": 0, "1 — Male (Laki-laki)": 1}
            wsex = st.sidebar.selectbox(
                'Sex (sex)',
                options=list(sex_map.keys()),
                help="Jenis kelamin pasien. Laki-laki memiliki risiko penyakit jantung lebih tinggi secara statistik"
            )
            sex = sex_map[wsex]

            age = st.sidebar.slider(
                'Age (age)',
                min_value=29, value=50, max_value=77, step=1,
                help="Usia pasien dalam tahun. Range dataset: 29–77 tahun"
            )

            # Urutan sesuai Sesi 13: cp, thalach, slope, oldpeak, exang, ca, thal, sex, age
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

        # Urutan sesuai training Sesi 13
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
