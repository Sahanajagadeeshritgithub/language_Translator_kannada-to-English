# Use the official Python image as a base
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
