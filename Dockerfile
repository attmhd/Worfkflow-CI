# Gunakan Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Salin isi folder MLProject ke dalam image
COPY MLProject/ /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install mlflow pandas scikit-learn imbalanced-learn matplotlib seaborn

# Jalankan script training
CMD ["mlflow", "run", "."]
