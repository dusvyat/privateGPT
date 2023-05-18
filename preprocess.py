import pandas as pd
from settings import COLUMNS_TO_DROP, COLUMN_RENAME_MAP, THRESHOLD_NULL_VALUES, FILL_NA_VALUES

from logging import getLogger

logger = getLogger(__name__)


class Preprocessor:
	def __init__(
			self,
			input_path: str='unprocessed_data/drug_context_data.csv',
			output_path: str="source_documents/drug_context_cleaned.csv"
	):

		self.input_path = input_path
		self.output_path = output_path

	def preprocess(
			self,
			threshold_null_values: float = 0.4,
			columns_to_drop: list=None,
			column_rename_map: dict=None,
			fill_na: bool = False
	):

		logger.info("Starting preprocessing data.")

		df = pd.read_csv(self.input_path, low_memory=False, encoding='utf-8')

		original_columns = list(df.columns)

		if columns_to_drop is not None:

			df.drop(
				columns=columns_to_drop, inplace=True
			)

		# remove columns with more than n% null values
		df.dropna(
			axis=1,
			thresh=(df.shape[0] * threshold_null_values),
			inplace=True
		)

		df.drop_duplicates(inplace=True)

		after_columns = list(df.columns)

		if fill_na:
			df.fillna(inplace=True, value='Data Not Available')

		# remove special characters
		df.replace(to_replace=r'[^a-zA-Z0-9 ]+', value='', regex=True, inplace=True)

		if column_rename_map is not None:

			df.rename(column_rename_map, axis=1, inplace=True)

		df["data_source"] = "U.S. Food and Drug Administration - FDA"

		logger.info('original columns: %s', original_columns)

		logger.info('after columns: %s', after_columns)

		logger.info("Finished preprocessing data... writing output to csv.")

		df.to_csv(self.output_path, index=False)


if __name__ == "__main__":

	preprocessor = Preprocessor()
	preprocessor.preprocess(
		columns_to_drop=COLUMNS_TO_DROP,
		column_rename_map=COLUMN_RENAME_MAP,
		threshold_null_values=THRESHOLD_NULL_VALUES,
		fill_na=FILL_NA_VALUES
	)