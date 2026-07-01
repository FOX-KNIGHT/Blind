# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install minimal system dependencies needed for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the production requirements file
COPY requirements_prod.txt ./

# Install python packages
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy all the application files into the container
COPY . .

# Define the port environment variable
ENV PORT=10000
EXPOSE $PORT

# Start the Flask-SocketIO app using Gunicorn and Eventlet
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app
