FROM apache/airflow:3.0.1-python3.10

USER airflow
RUN pip install \
    openai \
    pytesseract \
    pdf2image \
    pyspellchecker \
    Pillow \
    img2pdf \
    rapidfuzz \
    fuzzywuzzy \
    PyPDF2 \
    pymongo \
    opik \
    joblib \
    sentence-transformers \
    rapidocr-onnxruntime \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scipy==1.11.4 \
    scikit-learn==1.4.2 \
    scikit-image==0.22.0 \
    opencv-python-headless==4.8.1.78
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
