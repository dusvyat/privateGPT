import os
import glob
from typing import List
from dotenv import load_dotenv

import torch
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, LlamaCpp
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.docstore.document import Document
from langchain.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain.vectorstores import Chroma

# load environment variables
load_dotenv()

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
LLM_TYPE = os.environ.get('LLM_TYPE')
LLM_PATH = os.environ.get('LLM_PATH')

MAX_TOKENS_LIMIT = os.environ.get('MAX_TOKENS_LIMIT')
EMBEDDINGS_MODEL_TYPE = os.environ.get("EMBEDDINGS_MODEL_TYPE")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
        return loader.load()  # for csv, each row becomes a separate document
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    all_files = txt_files + pdf_files + csv_files

    for file_path in all_files:
        documents_per_file = load_single_document(file_path)
        for document in documents_per_file:
            if document.page_content is not None:
                yield document


def load_embeddings_model():
    match EMBEDDINGS_MODEL_TYPE:
        case "Llama":
            embedding_model = LlamaCppEmbeddings(model_path=EMBEDDINGS_MODEL_NAME, n_ctx=MAX_TOKENS_LIMIT)
        case "sentence-transformers":
            gpu_cpu = {'device': "gpu" if torch.cuda.is_available() else "cpu"}
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs=gpu_cpu)
        case _:
            raise ValueError(f"Model {EMBEDDINGS_MODEL_TYPE} not supported!")
    return embedding_model


def load_llm():
    callbacks = [StreamingStdOutCallbackHandler()]
    match LLM_TYPE:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=LLM_PATH, n_ctx=MAX_TOKENS_LIMIT, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=LLM_PATH, n_ctx=MAX_TOKENS_LIMIT, backend='gptj', callbacks=callbacks, verbose=False)
        case _:
            raise ValueError(f"Model {LLM_TYPE} not supported!")
    return llm


def load_chroma():
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model,
                  client_settings=CHROMA_SETTINGS)
