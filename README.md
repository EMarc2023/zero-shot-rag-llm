# Zero-Shot RAG-LLM: Amazon Review 2023 Dataset

## Project Overview
Developed a **production-grade, CPU-optimised, Retrieval Augmented Generation (RAG) system** using a subset (Gift Cards) of the **Amazon Review 2023 dataset**. The system leverages cutting-edge NLP techniques for **knowledge retrieval** and **document summarisation** with minimal training data requirements.

## Key Technologies
- **HuggingFace Transformers**: For pre-trained transformer models used in both retrieval and generation tasks.
- **FAISS**: Efficient similarity search for fast document retrieval at scale.
- **LangChain**: For orchestrating RAG-style workflows with language models.
- **Bart**: For summarising contextual data from what FAISS retrieves. Alternatively, truncate the context sentences using the API provided in the app.
- **TinyLlama**: For tokenisation (for both truncating context sentences and for inputting the prompt to the model) and question answering.
- **FastAPI**: Deployed as a production-ready web application for API access to the RAG system.
- **Docker**: Containerised the entire application for consistency and scalability.
- **CI/CD**: Fully automated deployment pipeline with tests, linting, and Docker integration for streamlined development and production flow.

## Project Features
- **Zero-Shot Retrieval Augmented Generation (RAG)**: Enables the model to retrieve relevant documents and generate insightful summaries without requiring task-specific training.
- **Production-Grade Deployment**: Dockerized API with FastAPI, ensuring scalability, modularity, and ease of deployment in cloud or on-prem environments.
- **Code Quality & Maintenance**:
  - **Pylint Score**: 10/10 (Ensures clean, maintainable code)
  - **CI/CD Pipeline**: Fully automated with unit tests, linting, and Docker integration. Build time ~11 minutes.
 
## Applications
This project can be applied to industries that require efficient AI-driven knowledge retrieval and document summarization, such as:
- **Enterprise solutions: Automating document analysis and retrieval.
- **Customer support: Answering customer queries based on large document datasets.
- **Content generation: Summarising or generating new content based on vast knowledge bases.

## Future Enhancements
- **Model performance metrics: Planned refinements to include standard evaluation metrics like BLEU and F1 score.
- **Optimisation: Further model fine-tuning and performance benchmarking to enhance retrieval accuracy and generation quality.
- **Frontend UI: Developing a React.js-based frontend for a user-friendly interface, allowing users to interact with the system
and visualise retrieved documents and generated summaries easily. This will provide a complete full-stack experience,
from backend API to frontend deployment.

## Access & Execution
- **Dockerisation**: The project is fully containerised. To run the application, simply clone the repository.
- For running the app, run the following (using Bash):

  uvicorn main:app --reload
  
- For building the Docker image, run the following (using Bash):
  
  docker build -t rag-llm .

  docker run -p 8000:8000 rag-llm
