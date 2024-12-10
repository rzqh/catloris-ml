from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from io import BytesIO
import json
import pandas as pd
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

# Inisialisasi FastAPI
app = FastAPI()

# Load model
model = load_model('food_image2.h5')

# Load dataset nutrisi dari JSON
with open('dataset_makanan.json', 'r') as f:
    dataset_makanan = json.load(f)

# Fungsi untuk memprediksi gambar makanan
def predict_image(image, model):
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    max_probability = np.max(prediction[0])
    
    class_labels = {0: 'ayam goreng krispi', 1: 'bakso', 2: 'burger', 3: 'kentang goreng', 
                    4: 'nasi goreng', 5: 'nasi padang', 6: 'nasi putih', 7: 'nugget', 
                    8: 'pizza', 9: 'rawon daging sapi', 10: 'rendang', 11: 'sate', 
                    12: 'seblak', 13: 'sop', 14: 'tempe goreng'}
    
    if max_probability >= 0.65:
        predicted_class = np.argmax(prediction)
        return class_labels[predicted_class], max_probability
    else:
        return "Makanan tidak dikenali", max_probability

# Fungsi untuk memprediksi nutrisi
def prediksi_nutrisi(nama_makanan):
    for label, makanan in dataset_makanan.items():
        if makanan['nama'] == nama_makanan:
            return makanan
    return "Nutrisi Makanan tidak ditemukan dalam dataset"

# Endpoint untuk prediksi gambar makanan
@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = load_img(BytesIO(contents), target_size=(224, 224))
        predicted_label, confidence = predict_image(image, model)

        if predicted_label == "Makanan tidak dikenali":
            raise HTTPException(status_code=400, detail={
                'prediksi': predicted_label,
                'confidence': float(confidence),
                'message': 'Tidak dapat memprediksi nutrisi karena makanan tidak dikenali.'
            })

        hasil_nutrisi = prediksi_nutrisi(predicted_label)

        if isinstance(hasil_nutrisi, dict):
            return {
                'prediksi makanan': predicted_label,
                'confidence': float(confidence),
                'nutrisi': hasil_nutrisi
            }
        else:
            raise HTTPException(status_code=404, detail={'error': hasil_nutrisi})

    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e)})

# Fungsi untuk menghitung BMI
def hitung_bmi(berat, tinggi):
    tinggi_m = tinggi / 100
    return berat / (tinggi_m ** 2)

# Fungsi untuk menentukan kategori BMI
def kategori_bmi(bmi):
    if bmi < 18.5: return "Kurus"
    elif 18.5 <= bmi < 25: return "Normal"
    elif 25 <= bmi < 30: return "Kelebihan Berat Badan"
    else: return "Obesitas"

# Fungsi untuk menentukan kategori BMI berdasarkan input
def tentukan_kategori_bmi(berat, tinggi):
    bmi = hitung_bmi(berat, tinggi)
    return bmi, kategori_bmi(bmi)

# Membaca dataset fisik dan nutrisi dari file CSV
file_kondisi_fisik = "Dataset Kondisi Fisik dan Kebutuhan Nutrisi.csv"
file_nutrisi_makanan = "Dataset Informasi Nutrisi Makanan.csv"

kondisi_fisik = pd.read_csv(file_kondisi_fisik)
nutrisi_makanan = pd.read_csv(file_nutrisi_makanan)

# Menambahkan kolom Key untuk penggabungan
kondisi_fisik['Key'] = 1
nutrisi_makanan['Key'] = 1
data = pd.merge(kondisi_fisik, nutrisi_makanan, on='Key', suffixes=('_x', '_y')).drop('Key', axis=1)

# Menentukan fitur dan target
features = ['Usia', 'Berat (kg)', 'Tinggi (cm)', 'Lemak (g)_x']
target = ['Kalori (kcal)_y', 'Protein (g)_y', 'Karbohidrat (g)_y', 'Lemak (g)_y']

# Preprocessing data
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), features)]
)

# Membuat pipeline untuk rekomendasi
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', NearestNeighbors(n_neighbors=5))])
pipeline.fit(data[features])

# Model input dari user
class UserData(BaseModel):
    usia: int
    berat: float
    tinggi: float
    lemak: float

# Endpoint untuk memberikan rekomendasi berdasarkan BMI dan data pengguna
@app.post('/rekomendasi')
def recommend(user_data: UserData):
    input_user = pd.DataFrame({
        'Usia': [user_data.usia],
        'Berat (kg)': [user_data.berat],
        'Tinggi (cm)': [user_data.tinggi],
        'Lemak (g)_x': [user_data.lemak]
    })

    bmi, kategori_bmi = tentukan_kategori_bmi(user_data.berat, user_data.tinggi)

    # Transformasi data pengguna dan mendapatkan rekomendasi
    input_transformed = pipeline['preprocessor'].transform(input_user)
    distances, indices = pipeline['model'].kneighbors(input_transformed)

    rekomendasi = data.iloc[indices[0]]

    result = {
        'bmi': round(bmi, 2),
        'kategori_bmi': kategori_bmi,
        'rekomendasi': rekomendasi[['Nama Makanan', 'Kalori (kcal)', 'Protein (g)_y', 'Karbohidrat (g)_y', 'Lemak (g)_y']].to_dict('records')
    }

    return result
