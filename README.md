# Zero-Shot RAG-LLM: Amazon Review 2023 Dataset

## Project Overview
- The application is Dockerised and deployed via FastAPI, ensuring scalability and modularity. It is production-ready for smaller to medium-scale deployments and can be further optimised for high-traffic, enterprise-level environments with more robust deployment options such as NGINX as a reverse proxy or using production-grade ASGI servers like Gunicorn.
- This application is a CPU-optimised Retrieval Augmented Generation (RAG) system using a subset (Gift Cards) of the Amazon Review 2023 dataset. The system leverages cutting-edge NLP techniques for efficient knowledge retrieval and document summarisation, achieving robust performance with minimal task-specific training data.

## Key Technologies

- **HuggingFace Transformers**: Pre-trained transformer models used in both retrieval and generation tasks.
- **FAISS**: Efficient similarity search for fast, scalable document retrieval, essential for large-scale applications.
- **LangChain**: Orchestrates RAG-style workflows, enabling dynamic interaction between retrieval and generation components.
- **BART**: Summarises contextual data from documents retrieved via FAISS. Alternatively, context sentences can be truncated using the API provided in the app.
- **TinyLlama**: Handles tokenisation for both truncating context sentences and inputting the prompt to the model for question answering.
- **FastAPI**: Deployed as a production-ready web application, providing an API interface to the RAG system.
- **Docker**: Containerises the entire application, ensuring consistency and scalability across environments.
- **CI/CD**: Fully automated deployment pipeline incorporating unit tests, linting, and Docker integration, streamlining both development and production workflows.

## Project Features

### Zero-Shot Retrieval Augmented Generation (RAG)
The system retrieves relevant documents and generates insightful summaries without requiring task-specific training, optimising for versatility and generalisation across domains.

### Production-Grade Deployment
The application is Dockerised and deployed via FastAPI, ensuring scalability, modularity, and ease of deployment in both cloud and on-premises environments.

## Code Quality & Maintenance

- **Pylint Score**: 10/10 (Ensures clean, maintainable code)
- **CI/CD Pipeline**: Fully automated with unit tests, linting, and Docker integration. **Build time**: ~11 minutes.

## Applications

This project is highly relevant for industries requiring efficient AI-driven knowledge retrieval and document summarisation, including:

- **Enterprise Solutions**: Automating document analysis and retrieval processes.
- **Customer Support**: Providing automated answers to customer queries based on extensive document datasets.
- **Content Generation**: Summarising or generating new content derived from vast knowledge bases.

## Future Enhancements

- **Model Performance Metrics**: Planned improvements to include standard evaluation metrics like BLEU and F1 scores for better benchmarking.
- **Optimisation**: Ongoing efforts to fine-tune the model and perform performance benchmarking to improve both retrieval accuracy and content generation quality.
- **Frontend UI**: Development of a React.js-based frontend to provide a user-friendly interface, enabling easy interaction with the system and visualisation of retrieved documents and generated summaries. This will complete the full-stack experience, from backend API to frontend deployment.

## Access & Execution

### Dockerisation

The project is fully containerised. To run the application, simply clone the repository.

#### Running the FastAPI app:

```bash
uvicorn main:app --reload
```

#### Building the Docker image:

```bash
docker build -t rag-llm .
```
```bash
docker run -p 8000:8000 rag-llm
```

