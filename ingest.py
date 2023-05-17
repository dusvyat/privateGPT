from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import time

from settings import SOURCE_DIRECTORY, load_documents, PERSIST_DIRECTORY, CHROMA_SETTINGS, load_embeddings_model


def run():

    start_time = time.time()

    # Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(load_documents(SOURCE_DIRECTORY))

    print(f"Loaded {len(list(load_documents(SOURCE_DIRECTORY)))} documents from {SOURCE_DIRECTORY}")
    print(f"Split into {len(texts)} chunks of text (max. 500 tokens each)")

    embedding_model = load_embeddings_model()

    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Fished creating vectorstore from documents.")
    print(f"Elapsed time: {round(elapsed_time/60)} minutes")


if __name__ == "__main__":
    run()
