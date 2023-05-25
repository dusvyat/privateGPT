import logging
import typer


from ingest import Ingestor
from preprocess import Preprocessor
from question_answer import QuestionAnswerer
from settings import OUTPUT_PATH, LOGGING, CONFIG_PATH, load_config
from logging import getLogger

#if time fix this
if LOGGING:

	logger = getLogger(__name__)

	logging.basicConfig(level=logging.INFO)

class Runner:
	"""Runner class to run the pipeline"""

	def __init__(self,config_name:str="writer_config.yml"):
		self.config_name = config_name
		self.config = load_config(self.config_name)

	def run(
			self,
			query:str=None,
			preprocess:bool=False,
			ingest:bool=False,
			save_output:bool=False
	):
		"""specify the pipeline to run using arguments,
		if using user input, specify the question to ask and when done enter 'exit' to save output,
		if not, specify the questions to ask in the queries file"""

		# preprocess data
		if preprocess:

			preprocessor = Preprocessor(config_name=self.config_name)
			preprocessor.preprocess()

		# embed data
		# ingest embeddings to vector db
		# persist vector db to disk

		if ingest:
			ingestor = Ingestor(config_name=self.config_name)
			ingestor.embeddings_to_vectordb()
		question_answerer = QuestionAnswerer(config_name=self.config_name)

		question_answerer.query(query)

		if save_output:
			question_answerer.output(OUTPUT_PATH)

		return question_answerer.results_df


if __name__ == "__main__":

	typer.run(Runner().run)
