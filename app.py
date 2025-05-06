# Copyright 2025 Elizabeth Marcellina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI app - a zero-shot RAG LLM app"""

import os
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from load_data_and_index import load_chunks, load_faiss_index

# Load environment variables from the .env file
load_dotenv()
chunk_path = os.getenv("CHUNK_PATH", "cleaned_chunk_texts.pkl")
faiss_path = os.getenv("FAISS_PATH", "faiss_index_from_cleaned_chunks")
MAX_LENGTH_SUMMARY = 50  # global variable
summariser_model = os.getenv("SUMMARISER_MODEL", "facebook/bart-base")
LLM_model = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Example of initialising variables at the module level (before their use)
chunk_texts = []
retriever = None  # pylint: disable=C0103
# bart_pipeline = None  # pylint: disable=C0103
bart_tokenizer = None  # pylint: disable=C0103
bart_model = None  # pylint: disable=C0103
llm = None  # pylint: disable=C0103
tokenizer = None  # pylint: disable=C0103


async def init():
    # pylint: disable=W0603
    """
    Initialise data, vector database, and transformers models
    """
    # Setup global variables
    global chunk_texts, retriever, bart_tokenizer, bart_model, llm, tokenizer

    # Setup multithreading
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(os.cpu_count())

    # Load serialised chunk texts
    print("Loading serialised chunk files...")
    chunk_texts = load_chunks(chunk_path)

    # Load FAISS index
    print("Loading FAISS index with LangChain wrapper...")
    retrieved_vector_db = load_faiss_index(faiss_path)
    retriever = retrieved_vector_db.as_retriever(search_kwargs={"k": 5})

    # Load BART summariser
    print("Loading Bart summariser...")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # Load LLM for answer generation
    print("Loading LLM...")
    torch.backends.mkldnn.enabled = True
    llm_model_name = LLM_model
    llm = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm = torch.compile(llm)  # Optimize model execution


def summarize_context_bart(context):
    """Summarizes the context using BART base."""
    # Tokenize the input context using BART tokenizer to handle the max token length
    inputs = bart_tokenizer(
        context, return_tensors="pt", truncation=True, max_length=1024
    )
    # Use the model to generate the summary
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=MAX_LENGTH_SUMMARY,
        min_length=40,
        num_beams=4,  # optional, for beam search; can adjust based on need
        early_stopping=True,  # optional, stop when the model generates the desired number of tokens
    )
    # Decode the summary into a human-readable format
    clean_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return clean_summary


def clean_text(text):
    """Remove non-alphanumeric characters (except spaces)."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_limited_context(context, max_length_summary_local):
    """Summarizes the context using truncation."""
    tokenized_context = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=max_length_summary_local,
    )
    decoded_context = tokenizer.decode(
        tokenized_context["input_ids"][0], skip_special_tokens=True
    )
    cleaned_decoded_context = clean_text(decoded_context)
    return cleaned_decoded_context


def summarise_or_truncate(query, method):
    """Retrieve documents."""
    results = retriever.invoke(query)
    full_context = " ".join([doc.page_content for doc in results])
    # Summarize
    if method == "bart":
        summary = summarize_context_bart(full_context)
    elif method == "truncate":
        summary = get_limited_context(full_context, MAX_LENGTH_SUMMARY)
    else:
        raise HTTPException(
            status_code=400, detail="Invalid method. Choose 'bart' or 'truncate'."
        )
    if not summary:
        raise HTTPException(status_code=400, detail="Summarisation failed.")
    return summary


def generate_response_with_llm(input_text):
    """Tokenise input text and generate an answer."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = llm.generate(
            inputs["input_ids"],
            max_length=200,  # limit the number of output tokens
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    cleaned_response = clean_llm_output(response)
    return cleaned_response


def clean_llm_output(response: str):
    """
    Clean the response from TinyLlama to return only the "Output:" part.
    """
    match = re.search(r"Output: (.*)", response)
    if match:
        return match.group(1).strip()  # Extract and clean the "Output:" part
    return response.strip()  # In case "Output:" is not found


class UserPromptRequest(BaseModel):
    """FastAPI class to take the user's input."""

    query: str = Field(
        ..., title="User query", description="Query to retrieve relevant documents."
    )
    method: str = Field(
        "bart",
        title="Summarisation method",
        description="Summarise context.",
        json_schema_extra={"enum": ["bart", "truncate"]},
    )
    prompt: str = Field(
        ..., title="User prompt", description="Custom instruction for LLM generation."
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup code."""
    await init()
    print("Models initialized successfully.")
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
origins = [
    "http://localhost:3000",  # Allow requests from your React frontend
    "http://localhost",  # For potential testing without the port
    "http://127.0.0.1:3000",  # Another common frontend address
    "http://127.0.0.1",  # For potential testing without the port
    "*",  # Be cautious with this in production; allows all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# The endpoints
# Home page
@app.get("/")
def home():
    """The app home page."""
    return {"message": "FastAPI FAISS Retrieval API is running!"}


# Endpoint to test FAISS retrieval
@app.get("/search/")
def search(query: str):
    """Search for relevant texts."""
    results = retriever.invoke(query)
    return {"query": query, "results": [doc.page_content for doc in results]}


# Endpoint to test FAISS retrieval and context
@app.get("/search_with_context/")
def search_with_context(
    query: str, method: str = Query("bart", enum=["bart", "truncate"])
):
    """Search for relevant texts and their summary or truncation."""
    summary = summarise_or_truncate(query, method)
    return {"query": query, "summary": summary}


# Endpoint to test app logic
@app.get("/answer_with_no_summary/")
def answer_with_no_summary(
    query: str, method: str = Query("bart", enum=["bart", "truncate"])
):
    """Mimicking the logic of getting an answer from the LLM for unit test."""
    input_text = f"Query: {query} Method: {method}"
    response = "Output of " + input_text
    return {"query": query, "response": response}


@app.get("/answer_with_default_prompt/")
def answer_with_default_prompt():
    """Get an answer from the LLM, using the default prompt."""
    default_prompt = "Hello"
    cleaned_response = generate_response_with_llm(default_prompt)
    return {"default_prompt": default_prompt, "cleaned_response": cleaned_response}


# Testing LLM response
@app.get("/answer/")
def answer(query: str, method: str = Query("bart", enum=["bart", "truncate"])):
    """Get an answer from the LLM using the summary only."""
    summary = summarise_or_truncate(query, method)
    cleaned_response = generate_response_with_llm(summary)
    return {"query": query, "summary": summary, "cleaned_response": cleaned_response}


# Endpoint to get a response, using the user's prompt
@app.post("/answer_with_user_prompts/")
def answer_with_user_prompts(request: UserPromptRequest):
    """Get an answer from the LLM, using the user's prompt."""
    summary = summarise_or_truncate(request.query, request.method)
    input_text = f"Prompt: {request.prompt} Query: {request.query}\
    Summary: {summary} (short bullet points only for the output) Output:"
    cleaned_response = generate_response_with_llm(input_text)

    return {
        "query": request.query,
        "summary": summary,
        "cleaned_response": cleaned_response,
    }


# Health check route
@app.get("/health_check")
async def health_check():
    """Health check route."""
    return {"status": "ok.", "Message": "The FastAPI app is up and running!"}
