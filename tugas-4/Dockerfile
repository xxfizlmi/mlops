# Gunakan base image Python
FROM python:3.9-slim

# Set environment variable untuk non-interaktif apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install make dan dependencies
RUN apt-get update && apt-get install -y make && apt-get clean

# Salin file requirements dan instal dependensi Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Salin seluruh kode proyek
COPY . .

# Perintah default untuk menjalankan Makefile
CMD ["make"]
