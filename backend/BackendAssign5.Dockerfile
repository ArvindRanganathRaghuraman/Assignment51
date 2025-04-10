FROM python:3.12.4

WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install sentence-transformers
RUN pip install --no-cache-dir sentence-transformers

# Copy app code and credentials
COPY . .
COPY starry-tracker-449020-f2-98e23064a0ed.json /app/starry-tracker-449020-f2-98e23064a0ed.json

# Handle pinecone cleanup and reinstallation
RUN pip uninstall -y pinecone pinecone-client pinecone-plugin-inference || true
RUN pip install --no-cache-dir pinecone

# Set Google credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/starry-tracker-449020-f2-98e23064a0ed.json"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
