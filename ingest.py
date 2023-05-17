from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import time

from settings import SOURCE_DIRECTORY, load_documents, PERSIST_DIRECTORY, CHROMA_SETTINGS, load_embeddings_model
from logging import getLogger

logger = getLogger(__name__)


class Ingestor:

    @staticmethod
    def split_documents():

        # Load documents and split in chunks
        logger.info(f"Loading documents from {SOURCE_DIRECTORY}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = load_documents(SOURCE_DIRECTORY)
        texts = text_splitter.split_documents(documents)

        logger.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
        logger.info(f"Split into {len(texts)} chunks of text (max. 500 tokens each)")

        return texts

    def embeddings_to_vectordb(self):

        start_time = time.time()

        # Load documents and split in chunks
        texts = self.split_documents()

        # Load embeddings model
        embedding_model = load_embeddings_model()

        # Create and store locally vectorstore
        db = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
        db.persist()
        db = None
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info("Fished creating vectorstore from documents.")
        logger.info(f"Elapsed time: {round(elapsed_time/60)} minutes")


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.embeddings_to_vectordb()
