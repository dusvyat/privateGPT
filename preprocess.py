import pandas as pd


def preprocess(input_path='unprocessed_data/drug_context_data.csv', output_path="source_documents/drug_context_cleaned.csv"):

	# remove columns with more than 40% null values

	df = pd.read_csv(input_path, low_memory=False, encoding='utf-8')

	original_columns = list(df.columns)


	df.drop(
		columns=['openfda_spl_set_id', 'openfda_product_ndc', 'openfda_spl_id', 'openfda_package_ndc', 'version',
		         'set_id','openfda_unii','spl_unclassified_section','openfda_application_number','effective_time'], inplace=True
	)
	df.dropna(
		axis=1, thresh=df.shape[0] * 0.4, inplace=True
	)

	df.drop_duplicates(inplace=True)

	after_columns = list(df.columns)

	df.fillna(inplace=True, value='Data Not Available')

	# remove special characters
	df.replace(to_replace=r'[^a-zA-Z0-9 ]+', value='', regex=True, inplace=True)

	col_rename_dict = {
		'package_label_principal_display_panel': 'product_package_label',
		'openfda_manufacturer_name': 'product_manufacturer_name',
		'openfda_product_type':"type_of_product",
		'openfda_route': "how_to_use_product",
		'purpose': "intended_purpose_of_product",
		'openfda_generic_name':"generic_name",
		'openfda_brand_name':"brand_name",
		'openfda_substance_name': "substance_name",
		"spl_product_data_elements":"full_ingredients",
		"keep_out_of_reach_of_children":"keep_out_of_reach_of_children_warning",
		"warnings":"product_warnings_and_cautions",
		"id":"FDA_product_id",
	}

	df.rename(col_rename_dict, axis=1, inplace=True)

	df["data_source"] = "U.S. Food and Drug Administration - FDA"

	print('original columns: ', original_columns)

	print('after columns: ', after_columns)

	df.to_csv(output_path, index=False)


if __name__ == "__main__":

	preprocess()
