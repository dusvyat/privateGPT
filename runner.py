import logging
import typer


from ingest import Ingestor
from preprocess import Preprocessor
from question_answer import QuestionAnswerer
from settings import COLUMNS_TO_DROP, COLUMN_RENAME_MAP, THRESHOLD_NULL_VALUES, FILL_NA_VALUES, QUERIES, OUTPUT_PATH,TOGGLE_LOGGING
from logging import getLogger

if TOGGLE_LOGGING:

	logger = getLogger(__name__)

	logging.basicConfig(level=logging.INFO)


def runner(
		query:str=None,
		user_input:bool=False,
		preprocess:bool=False,
		ingest:bool=False,
		save_output:bool=False
):
	"""specify the pipeline to run using arguments,
	if using user input, specify the question to ask and when done enter 'exit' to save output,
	if not, specify the questions to ask in the queries file"""

	# preprocess data
	if preprocess:

		preprocessor = Preprocessor()
		preprocessor.preprocess(
			columns_to_drop=COLUMNS_TO_DROP,
			column_rename_map=COLUMN_RENAME_MAP,
			threshold_null_values=THRESHOLD_NULL_VALUES,
			fill_na=FILL_NA_VALUES
		)

	# embed data
	# ingest embeddings to vector db
	# persist vector db to disk

	if ingest:
		ingestor = Ingestor()
		ingestor.embeddings_to_vectordb()
	question_answerer = QuestionAnswerer()

	if user_input and query is None:
		question_answerer.query()
	else:
		question_answerer.query(query)

	if save_output:
		question_answerer.output(OUTPUT_PATH)

	return question_answerer.results_df


if __name__ == "__main__":
	typer.run(runner)


