# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install minimal system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the production requirements file
COPY requirements_prod.txt ./

# Install python packages
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy all the application files into the container
COPY . .

# Define the port environment variable
ENV PORT=5000

# Run the application using gunicorn with custom eventlet worker class
CMD gunicorn --workers 1 --worker-class custom_worker.CustomEventletWorker --timeout 120 --bind 0.0.0.0:${PORT:-7860} app:app
