FROM apache/airflow:3.0.1-python3.10

USER airflow
RUN pip install openai pytesseract pdf2image pyspellchecker Pillow img2pdf rapidfuzz fuzzywuzzy PyPDF2 pymongo opik scikit-learn scikit-image joblib sentence-transformers easyocr opencv-python-headless
USER airflow

USER root
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER root
