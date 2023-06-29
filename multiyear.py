import os
import main
import pandas as pd


# Create a function to analyze multiyear weights
def combine_multiyear_weights(file_identifier):
    # Define the base directory where the dataframes are stored
    base_directory = f"Multiyear/{file_identifier}"

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Loop through the directories in the base directory
    for directory in os.listdir(base_directory):
        # Read the weights_with_tokens dataframe in the current directory
        filepath = os.path.join(base_directory, directory, "weights_with_tokens")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found for {directory}")  # Skip directories without the desired file
        weight_tokens_df = pd.read_csv(filepath)

        # Rename the 'weights' column to include the year
        weights_col_name = f"Weights_{directory}"
        weight_tokens_df = weight_tokens_df.rename(columns={"Weights": weights_col_name})

        # Merge the current dataframe with the combined dataframe
        if combined_df.empty:
            combined_df = weight_tokens_df
        else:
            combined_df = pd.merge(combined_df, weight_tokens_df, on="Token", how="outer")

    # Sort the columns based on the year
    combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)
    combined_df.to_csv(base_directory + '/combined_weights', index=False)


# Analyze the combined weights
def analyze_multiyear_weights(file_identifier):
    multiyear_weights = pd.read_csv(f'Multiyear/{file_identifier}/combined_weights')
    print(f'There are {len(multiyear_weights)} rows in multiyear')

    # Remove rows with NaN values in any column
    multiyear_weights = multiyear_weights.dropna()
    print(f'There are {len(multiyear_weights)} non-na rows')
    print(multiyear_weights['Token'])


# Run the multiyear analysis
def run_multiyear_analysis(keyword, augment_predictions, fifty_fifty, ones_accuracy, second_keyword, lambda_value,
                           minimum_appearances_prevalence):
    # For every year between 2012 and 2022
    for year in range(2012, 2023):
        main.main(f'Datasets/one_bio_per_year/one_bio_per_year_{year}.csv',
                     keyword=keyword,
                     augment_predictions=augment_predictions,
                     fifty_fifty=fifty_fifty,
                     ones_accuracy=ones_accuracy,
                     second_keyword=second_keyword,
                     lambda_value=lambda_value,
                     minimum_appearances_prevalence=minimum_appearances_prevalence,
                     multiyear=True,  # For this file at least
                     save_results=True)


# I put everything in here because it's not worth it to make a new method
if __name__ == '__main__':
    keyword = 'nsfw'
    augment_predictions = True
    fifty_fifty = False  # if fifty_fifty, it shouldn't be one's accuracy
    ones_accuracy = False  # If ones_accuracy, there shouldn't be a second keyword
    second_keyword = 'porn'
    lambda_value = 1e-05
    minimum_appearances_prevalence = 1  # Should always be prevalence, not minimum appearances

    # run_multiyear_analysis(keyword, augment_predictions, fifty_fifty, ones_accuracy, second_keyword, lambda_value, minimum_appearances_prevalence)

    combine_multiyear_weights('nsfw_porn_1prevalence_1e-05lambda_augment')

    analyze_multiyear_weights('nsfw_porn_1prevalence_1e-05lambda_augment')