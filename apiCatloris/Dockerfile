# Gunakan base image Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Instal dependensi sistem
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Ekspos port 8000
EXPOSE 8000

# Gunakan uvicorn untuk menjalankan server dan pastikan mendengarkan di host 0.0.0.0
CMD ["uvicorn", "app1:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

