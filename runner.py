import logging
from typing import Union

from ingest import Ingestor
from preprocess import Preprocessor
from question_answer import query
from settings import COLUMNS_TO_DROP, COLUMN_RENAME_MAP, THRESHOLD_NULL_VALUES, FILL_NA_VALUES, QUERIES, OUTPUT_PATH,TOGGLE_LOGGING
from logging import getLogger
from datetime import datetime

import os

import pandas as pd

logging.basicConfig(level=logging.INFO)

logger = getLogger(__name__) if TOGGLE_LOGGING else None


def runner(
		queries: Union[list[str], str],
		preprocess:bool=False,
		ingest:bool=False,
		output_answer:bool=True
):

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

	answer = query(queries)

	if not output_answer:
		return answer

	output(answer)


def output(df: pd.DataFrame):
	# output results
	date_time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

	df.to_csv(f"data/outputs/answers-{date_time_now}-output.csv", index=False)


if __name__ == "__main__":
	runner(QUERIES)


