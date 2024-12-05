# Gunakan TensorFlow GPU image sebagai base image
FROM tensorflow/tensorflow:latest-gpu

# Set working directory
WORKDIR /app

# Copy semua file aplikasi ke dalam container
COPY . /app

# Instal dependensi dari requirements.txt
RUN pip install --ignore-installed --no-cache-dir -r requirements.txt

# Ekspos port yang digunakan oleh Flask
EXPOSE 5000

# Atur variabel lingkungan untuk Flask
ENV FLASK_RUN_PORT=5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
