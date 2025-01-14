from typing import Union
from collections import defaultdict

import pandas as pd
from langchain.chains import RetrievalQA
from settings import OUTPUT_PATH, load_config
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class QuestionAnswerer:

    def __init__(self, config_name: str = "writer_config.yml"):

        self.config = load_config(config_name)
        llm = self.config.load_llm()
        retriever = self.config.get_retriever()

        if self.config.llm_type == 'HFH':
            from langchain import PromptTemplate
            prompt = PromptTemplate.from_file(template_file=self.config.prompt_template_file, input_variables=self.config.prompt_input_variables)
            chain_type_kwargs = {"prompt": prompt}
            self.qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )

        else:
            self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

        self.output_data = defaultdict(list)

    @property
    def results_df(self):
        return pd.DataFrame(self.output_data)

    def user_query(self):
        while True:
            user_input = input("\n Please enter a question: ")
            if user_input == "exit":
                return
            self._query(user_input)

    def _multi_query(self,input_queries:list[str]):
        #todo investigate sig11 error

        """
        Iterate through a list of queries and log the results.
        """

        for input_query in input_queries:
            self._query(input_query)

    def _query(self, input_query:str):

        res = self.qa(input_query)

        logger.info("\n\n> Question:")
        logger.info(input_query)
        logger.info("\n> Answer:")
        logger.info(res['result'])

        self.output_data["question"].append(input_query)
        self.output_data["answer"].append(res['result'])

        for idx,document in enumerate(res['source_documents']):
            sources = defaultdict(list)
            logger.info("\n> " + document.metadata["source"] + ":")
            logger.info(document.page_content)
            sources['sources_document'].append(document.metadata["source"])
            sources['sources_row'].append(document.metadata.get("row", None))
            sources['sources_content'].append(document.page_content)

        self.output_data["source_documents"].append(dict(sources))


    def query(self, input_query: Union[list[str], str] = None):

        """Query the chroma database for an answer to a question."""

        # use input via if no query is given
        if input_query is None:
            self.user_query()
            return
        # Get the answer from the chain
        if isinstance(input_query, list):
            return self._multi_query(input_query)
        elif isinstance(input_query, str):
            return self._query(input_query)
        else:
            raise TypeError("query must be a string or list of strings.")

    def output(self,output_path:str=OUTPUT_PATH):
        # output results
        date_time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.results_df.to_csv(output_path / f"answers-{date_time_now}-output.csv", index=False)


if __name__ == "__main__":

    question_answerer = QuestionAnswerer(config_name="hf_config.yml")
    question_answerer.query("what product to use to treat a sore throat")


    print('finished')
