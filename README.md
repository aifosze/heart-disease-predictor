# 🌸 ML Portfolio - Iris & Heart Disease Prediction

Aplikasi web interaktif menggunakan **Streamlit** untuk memprediksi jenis Iris dan risiko penyakit jantung dengan Machine Learning. Dilengkapi dengan interface yang user-friendly dan support CSV upload.

## ✨ Fitur Utama

- 🏠 **Homepage Interaktif** - Tampilan welcome dengan overview aplikasi
- 🔍 **Iris Species Prediction** - Klasifikasi 3 jenis iris (Setosa, Versicolor, Virginica)
- ❤️ **Heart Disease Prediction** - Prediksi risiko penyakit jantung
- 📤 **CSV Upload Support** - Upload file dataset untuk batch prediction
- 🎚️ **Interactive Sliders** - Input data manual dengan slider interaktif
- ⚡ **Real-time Results** - Hasil prediksi instant setelah submit
- 📊 **Data Visualization** - Tampilan data dan hasil dalam format yang mudah dipahami

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Model**: Scikit-learn (Pre-trained models)
- **Data Processing**: Pandas
- **Model Storage**: Pickle

## 📋 Requirements

```
streamlit
pandas
scikit-learn
pickle
```

## ⚙️ Installation

1. **Clone atau download project ini**
   ```bash
   cd sesi_15_live
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Atau install manual:
   ```bash
   pip install streamlit pandas scikit-learn
   ```

3. **Pastikan file model tersedia**
   - `generate_iris.pkl` - Pre-trained Iris classification model

## 🚀 Cara Menjalankan

1. **Buka terminal di folder project**

2. **Jalankan command**
   ```bash
   streamlit run iris_prediction.py
   ```

3. **Buka browser** - Aplikasi akan otomatis terbuka di `http://localhost:8501`

## 📖 User Guide

### 1. Home Page
- Lihat overview aplikasi
   - Daftar fitur yang tersedia
- Panduan cara menggunakan aplikasi

### 2. Iris Species Prediction
**Input Data:**
- **Mode Upload CSV**: Upload file CSV dengan 4 kolom (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
- **Mode Manual Input**: Gunakan slider untuk input 4 parameter:
  - Sepal Length: 4.3 - 10.0 cm
  - Sepal Width: 2.0 - 5.0 cm
  - Petal Length: 1.0 - 9.0 cm
  - Petal Width: 0.1 - 5.0 cm

**Output Prediksi:**
- Iris-setosa
- Iris-versicolor
- Iris-virginica

### 3. Heart Disease Prediction
Input parameter kesehatan Anda dan dapatkan prediksi risiko penyakit jantung.

## 📁 Project Structure

```
sesi_15_live/
├── iris_prediction.py          # Main application
├── generate_iris.pkl            # Pre-trained Iris model
├── requirements.txt             # Project dependencies
└── README.md                    # Documentation
```

## 🎯 Dataset Information

### Iris Dataset
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/uciml/iris)
- **Samples**: 150
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Features**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)

## 📊 Model Performance

Model telah dilatih dengan dataset Iris yang terstandar dan menghasilkan akurasi tinggi dalam klasifikasi.

## 💡 Tips

- Gunakan nilai default pada slider sebagai referensi
- Untuk batch prediction, siapkan CSV dengan format yang sesuai
- Hasil prediksi ditampilkan dengan loading animation untuk user experience yang baik

## 🔄 Model Update

Untuk update model, latih ulang dengan dataset baru dan simpan dengan format pickle:

```python
import pickle
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("generate_iris.pkl", "wb") as file:
    pickle.dump(model, file)
```

## ⚠️ Limitations

- Model hanya menerima 4 fitur numerik sesuai standar Iris dataset
- Prediksi hanya bernilai informatif untuk data dalam range dataset training
- Akurasi prediksi bergantung pada kualitas input data

## 📞 Support

Untuk pertanyaan atau bug report, silakan buat issue di repository ini.

## 📝 License

Proyek ini tersedia untuk keperluan edukasi dan pembelajaran.

---

**Created with ❤️ using Streamlit**

