# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV, ffmpeg, and other required packages
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install the Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire app directory into the container
COPY . /app/

# Expose the ports that Flask (5000) and FastAPI (8000) will run on
EXPOSE 5000
EXPOSE 8000

# Set environment variables for Flask and FastAPI
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Run the application using a script that starts Flask, FastAPI, and any other background tasks
CMD ["sh", "-c", "python app.py"]
