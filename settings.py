import os
import glob
from functools import lru_cache
from typing import List
from pathlib import Path
from pydantic import BaseModel
import yaml

import torch
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, LlamaCpp, Writer, HuggingFaceHub
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.docstore.document import Document
from langchain.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain.vectorstores import Chroma

# Load the environment variables
load_dotenv()

LOGGING = True

# Define the paths

PACKAGE_ROOT = Path(__file__).parent.resolve()
OUTPUT_PATH = PACKAGE_ROOT / "data" / "output"
QUERY_PATH = PACKAGE_ROOT / "data" / "queries"
CONFIG_PATH = PACKAGE_ROOT / "data" / "config"

INPUT_QUERIES_FILE = 'queries.txt'
PERSIST_DIRECTORY = (PACKAGE_ROOT / 'db').as_posix()
SOURCE_DIRECTORY = PACKAGE_ROOT / 'source_documents'

check_path_exists = lambda path: os.makedirs(path, exist_ok=True)

# create output and query directory if they don't exist
check_path_exists(OUTPUT_PATH)
check_path_exists(QUERY_PATH)

# Define the Chroma settings

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


class Config(BaseModel):
    input_queries_file: str = 'queries.txt'
    embeddings_model_type: str = 'sentence-transformers'
    embeddings_model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    llm_type: str = 'GPT4All'
    llm_path: str = 'models/ggml-gpt4all-j-v1.3-groovy.bin'
    max_token_limit: int = 1024
    columns_to_drop: List[str] = []
    column_rename_map: dict = {}
    threshold_null_values: float = 0.4
    fill_na_values: bool = False
    queries: List[str] = [] # list of queries to run or can use the input_queries_file

    @classmethod
    def parse_yaml(cls, path: str):
        with open(path) as f:
            config = yaml.safe_load(f)

        return cls.parse_obj(config)

    def load_embeddings_model(self):
        match self.embeddings_model_type:
            case "Llama":
                embedding_model = LlamaCppEmbeddings(model_path=self.embeddings_model_name, n_ctx=self.max_token_limit)
            case "hugging-face-sentence-transformers":
                model_kwargs = {'device': "gpu" if torch.cuda.is_available() else "cpu"}
                embedding_model = HuggingFaceEmbeddings(model_name=self.embeddings_model_name, model_kwargs=model_kwargs)
            case _:
                raise ValueError(f"{self.embeddings_model_type} embeddings model is not supported!")
        return embedding_model

    #todo create separate config object for llm model parameters
    def load_llm(self):
        callbacks = [StreamingStdOutCallbackHandler()]
        match self.llm_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=self.llm_path, n_ctx=self.max_token_limit, callbacks=callbacks, verbose=False)
            case "GPT4All":
                llm = GPT4All(model=self.llm_path, n_ctx=self.max_token_limit, backend='gptj', callbacks=callbacks,verbose=False)
            case "Writer":
                llm = Writer(model_id=self.llm_path, callbacks=callbacks, max_tokens=self.max_token_limit, verbose=False)
            case "HuggingFaceHub":
                model_kwargs = {'max_tokens': self.max_token_limit}
                llm = HuggingFaceHub(repo_id=self.llm_path, task='text2text-generation', model_kwargs=None, callbacks=callbacks, verbose=False)
            case _:
                raise ValueError(f"The {self.llm_type} of LLM is not supported!")
        return llm

    def load_chroma(self):
        return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=self.load_embeddings_model(),
                      client_settings=CHROMA_SETTINGS)

    def get_retriever(self):
        db = self.load_chroma()
        retriever = db.as_retriever()
        return retriever


def load_queries(file_path: str) -> List[str]:
    with open(file_path) as f:
        queries = f.readlines()

    return queries


QUERIES = load_queries(QUERY_PATH / INPUT_QUERIES_FILE)

@lru_cache()
def load_config(path) -> Config:
    return Config.parse_yaml(path=CONFIG_PATH / path)



@lru_cache()
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

    documents = []
    for file_path in all_files:
        documents_per_file = load_single_document(file_path)
        for document in documents_per_file:
            if document.page_content is not None:
                documents.append(document)

    return documents
