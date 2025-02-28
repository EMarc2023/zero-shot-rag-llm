# Use a base Python image
FROM python:3.10.12

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Set the environmental variables
ENV CHUNK_PATH=cleaned_chunk_texts.pkl
ENV FAISS_PATH=faiss_index_from_cleaned_chunks
ENV SUMMARISER_MODEL=facebook/bart-base
ENV LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
