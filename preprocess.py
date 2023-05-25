import pandas as pd
from settings import load_config

from logging import getLogger

logger = getLogger(__name__)


class Preprocessor:
	def __init__(
			self,
			input_path: str='unprocessed_data/drug_context_data.csv',
			output_path: str="source_documents/drug_context_cleaned.csv",
			config_name: str = "writer_config.yml"
	):

		self.input_path = input_path
		self.output_path = output_path
		self.config = load_config(config_name)

	def preprocess(
			self
	):

		logger.info("Starting preprocessing data.")

		df = pd.read_csv(self.input_path, low_memory=False, encoding='utf-8')

		original_columns = list(df.columns)

		if self.config.columns_to_drop is not None:

			df.drop(
				columns=self.config.columns_to_drop, inplace=True
			)

		# remove columns with more than n% null values
		df.dropna(
			axis=1,
			thresh=(df.shape[0] * self.config.threshold_null_values),
			inplace=True
		)

		df.drop_duplicates(inplace=True)

		after_columns = list(df.columns)

		if self.config.fill_na_values:
			df.fillna(inplace=True, value='Data Not Available')

		# remove special characters
		df.replace(to_replace=r'[^a-zA-Z0-9 ]+', value='', regex=True, inplace=True)

		if self.config.column_rename_map is not None:

			df.rename(self.config.column_rename_map, axis=1, inplace=True)

		df["data_source"] = "U.S. Food and Drug Administration - FDA"

		logger.info('original columns: %s', original_columns)

		logger.info('after columns: %s', after_columns)

		logger.info("Finished preprocessing data... writing output to csv.")

		df.to_csv(self.output_path, index=False)


if __name__ == "__main__":

	preprocessor = Preprocessor()
	preprocessor.preprocess()