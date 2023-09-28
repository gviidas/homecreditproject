FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Install system libraries required for LightGBM
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean &&

# Install necessary packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy prediction script
COPY predict.py /app/predict.py

# Copy machine learning models and other necessary files
COPY lgbm_undersampled.pkl /app/lgbm_undersampled.pkl


# Run the Flask app
CMD ["python", "predict.py"]
