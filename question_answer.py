from typing import Union
from collections import defaultdict

import pandas as pd
from langchain.chains import RetrievalQA
from settings import load_llm, load_chroma, QUERIES
import logging


logger = logging.getLogger(__name__)


def _load_chroma_chain():
    llm = load_llm()
    db = load_chroma()
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def _multi_query(input_queries:list[str], qa:RetrievalQA, output_data:dict):

    """
    Iterate through a list of queries.txt and log the results.
    """

    output_dfs = []

    for input_query in input_queries:

        output_dfs.append(_query(qa, input_query, output_data))

    return pd.concat(output_dfs)


def _query(qa:RetrievalQA, input_query:str, output_data:dict):

    res = qa(input_query)

    # logger.info("\n\n> Question:")
    # logger.info(input_query)
    # logger.info("\n> Answer:")
    # logger.info(res['result'])

    output_data["question"].append(input_query)
    output_data["answer"].append(res['result'])

    for idx,document in enumerate(res['source_documents']):
        sources = defaultdict(list)
        # logger.info("\n> " + document.metadata["source"] + ":")
        # logger.info(document.page_content)
        sources['sources_document'].append(document.metadata["source"])
        sources['sources_row'].append(document.metadata.get("row", None))
        sources['sources_content'].append(document.page_content)
    output_data["source_documents"].append(sources)

    return pd.DataFrame.from_dict(output_data)


def query(input_query: Union[list[str], str] = None):

    """Query the chroma database for an answer to a question."""

    qa = _load_chroma_chain()

    output_data = defaultdict(list)

    # use input via if no query is given
    if input_query is None:
        while True:
            input_query = input("\nEnter a query: ")
            if input_query == "exit":
                break

    # Get the answer from the chain
    if isinstance(input_query, list):
        return _multi_query(input_query, qa, output_data)
    elif isinstance(input_query, str):
        return _query(qa, input_query, output_data)
    else:
        raise TypeError("query must be a string or list of strings.")

if __name__ == "__main__":
    df=query(QUERIES)
