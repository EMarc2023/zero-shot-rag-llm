# Zero-Shot RAG-LLM: Amazon Review 2023 Dataset

## Project Overview
Developed a **production-grade Retrieval Augmented Generation (RAG) system** using the **Amazon Review 2023 dataset**. The system leverages cutting-edge NLP techniques for **knowledge retrieval** and **document summarization** with minimal training data requirements.

## Key Technologies
- **HuggingFace Transformers**: For pre-trained transformer models used in both retrieval and generation tasks.
- **FAISS**: Efficient similarity search for fast document retrieval at scale.
- **LangChain**: For orchestrating RAG-style workflows with language models.
- **FastAPI**: Deployed as a production-ready web application for API access to the RAG system.
- **Docker**: Containerized the entire application for consistency and scalability.
- **CI/CD**: Fully automated deployment pipeline with tests, linting, and Docker integration for streamlined development and production flow.

## Project Features
- **Zero-Shot Retrieval Augmented Generation (RAG)**: Enables the model to retrieve relevant documents and generate insightful summaries without requiring task-specific training.
- **Production-Grade Deployment**: Dockerized API with FastAPI, ensuring scalability, modularity, and ease of deployment in cloud or on-prem environments.
- **Code Quality & Maintenance**:
  - **Pylint Score**: 10/10 (Ensures clean, maintainable code)
  - **CI/CD Pipeline**: Fully automated with unit tests, linting, and Docker integration. Build time ~11 minutes.

## Access & Execution
- **Dockerization**: The project is fully containerized. To run the application, simply clone the repository and build the Docker image:
  *** docker build -t rag-llm .
  *** docker run -p 8000:8000 rag-llm
