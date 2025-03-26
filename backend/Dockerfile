# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy the credentials file into the container
COPY starry-tracker-449020-f2-98e23064a0ed.json /app/starry-tracker-449020-f2-98e23064a0ed.json

# Set environment variable inside the container
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/starry-tracker-449020-f2-98e23064a0ed.json"

# Expose the necessary ports
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
