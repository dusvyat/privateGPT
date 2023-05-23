import os
import glob
from typing import List

import torch
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All, LlamaCpp
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.docstore.document import Document
from langchain.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain.vectorstores import Chroma

from pathlib import Path

#todo replace with config file

load_dotenv()

TOGGLE_LOGGING = True

PACKAGE_ROOT = Path(__file__).parent.resolve()
OUTPUT_PATH = PACKAGE_ROOT / "data" / "output"
QUERY_PATH = PACKAGE_ROOT / "data" / "queries"

INPUT_QUERIES_FILE = 'queries.txt'
PERSIST_DIRECTORY = (PACKAGE_ROOT / 'db').as_posix()
SOURCE_DIRECTORY = PACKAGE_ROOT / 'source_documents'
EMBEDDINGS_MODEL_TYPE = 'sentence-transformers'
EMBEDDINGS_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
LLM_TYPE = 'GPT4All'
LLM_PATH = 'models/ggml-gpt4all-j-v1.3-groovy.bin'

MAX_TOKENS_LIMIT = 1024 # max token limit for LLM and embeddings

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


COLUMNS_TO_DROP = [
    'openfda_spl_set_id',
    'openfda_product_ndc',
    'openfda_spl_id',
    'openfda_package_ndc',
    'version',
    'set_id',
    'openfda_unii',
    'spl_unclassified_section',
    'openfda_application_number',
    'effective_time'
]

COLUMN_RENAME_MAP = {
    'package_label_principal_display_panel': 'product_package_label',
    'openfda_manufacturer_name': 'product_manufacturer_name',
    'openfda_product_type': "type_of_product",
    'openfda_route': "how_to_use_product",
    'purpose': "intended_purpose_of_product",
    'openfda_generic_name': "generic_name",
    'openfda_brand_name': "brand_name",
    'openfda_substance_name': "substance_name",
    "spl_product_data_elements": "full_ingredients",
    "keep_out_of_reach_of_children": "keep_out_of_reach_of_children_warning",
    "warnings": "product_warnings_and_cautions",
    "id": "FDA_product_id",

}

THRESHOLD_NULL_VALUES = 0.4
FILL_NA_VALUES = False


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


def load_embeddings_model():
    match EMBEDDINGS_MODEL_TYPE:
        case "Llama":
            embedding_model = LlamaCppEmbeddings(model_path=EMBEDDINGS_MODEL_NAME, n_ctx=MAX_TOKENS_LIMIT)
        case "sentence-transformers":
            model_kwargs = {'device': "gpu" if torch.cuda.is_available() else "cpu"}
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs = model_kwargs)
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
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=load_embeddings_model(),
                  client_settings=CHROMA_SETTINGS)


def get_retriever():
    db = load_chroma()
    retriever = db.as_retriever()
    return retriever


def load_queries(file_path: str) -> List[str]:
    with open(file_path) as f:
        queries = f.readlines()

    return queries


QUERIES = load_queries(QUERY_PATH / INPUT_QUERIES_FILE)
