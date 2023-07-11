import os
import matplotlib.pyplot as plt
import main
import pandas as pd
import numpy as np


# Create a function to analyze multiyear weights
def combine_multiyear_weights(file_identifier, start_year=0, normalize=True):
    # Define the base directory where the dataframes are stored
    base_directory = f"Multiyear/{file_identifier}"

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Loop through the directories in the base directory
    for directory in os.listdir(base_directory):
        # Read the weights_with_tokens dataframe in the current directory
        filepath = os.path.join(base_directory, directory, "weights_with_tokens")
        if not os.path.isfile(filepath):
            continue  # Skip directories without the desired file

        # If start_year is not none and the year is before start year, skip it
        if int(directory) < start_year:
            continue

        # Load the dataframe
        weight_tokens_df = pd.read_csv(filepath)

        # Remove any 0 weights
        weight_tokens_df = weight_tokens_df[weight_tokens_df["Weights"] != 0]

        # Normalize if desired
        if normalize:
            # Calculate the mean and standard deviation of the "Weights" column
            mean = weight_tokens_df['Weights'].mean()
            std = weight_tokens_df['Weights'].std()

            # Normalize the "Weights" column
            weight_tokens_df['Weights'] = (weight_tokens_df['Weights'] - mean) / std

        # Rename the 'weights' column to include the year
        weights_col_name = f"Weights_{directory}"
        weight_tokens_df = weight_tokens_df.rename(columns={"Weights": weights_col_name})

        # Merge the current dataframe with the combined dataframe
        if combined_df.empty:
            combined_df = weight_tokens_df
        else:
            combined_df = pd.merge(combined_df, weight_tokens_df, on="Token", how="outer")

    # Sort the columns based on the year and save the file
    combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)
    combined_df.to_csv(base_directory + '/combined_weights', index=False)


# Analyze the combined weights
def analyze_multiyear_weights(file_identifier, degree_fit=1):
    # Get the dataframe
    base_directory = f"Multiyear/{file_identifier}"
    multiyear_weights = pd.read_csv(f'{base_directory}/combined_weights')
    print(f'There are {len(multiyear_weights)} rows in multiyear')

    # Remove rows with NaN values in any column
    multiyear_weights = multiyear_weights.dropna()
    print(f'There are {len(multiyear_weights)} non-na rows')

    # Save the non-NaN dataframe
    multiyear_weights.to_csv(base_directory + f'/combined_weights_nonNaN_{degree_fit}', index=False)

    # Convert column values to float64 and get rid of NaNs
    multiyear_weights.iloc[:, 1:] = multiyear_weights.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Extract the years from column names and calculate the slope
    years = [int(col.split('_')[1]) for col in multiyear_weights.columns if col.startswith('Weights_')]
    slopes = []
    for index, row in multiyear_weights.iterrows():
        weights = row[1:].astype(float)
        slope = np.polyfit(years, weights, degree_fit)[0]
        slopes.append(slope)
    multiyear_weights['Slope'] = slopes

    # Sort by the weight of Slope
    sorted_multiyear_weights = multiyear_weights.sort_values('Slope', ascending=False)

    # Save the dataframe
    sorted_multiyear_weights.to_csv(base_directory + f'/sloped_combined_weights_{degree_fit}', index=False)


# Run the multiyear analysis
def run_multiyear_analysis(keyword, augment_predictions, fifty_fifty, ones_accuracy, second_keyword, lambda_value,
                           minimum_appearances_prevalence, default_amount, max_training_size):
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
                  save_results=True,
                  default_amount=default_amount,
                  max_training_size=max_training_size)


# Plots the top and bottom weights, and saves them in the respective Multiyear folder
def plot_top_and_bottom_weights(file_identifier, degree_fit):
    # Get the dataframe
    base_directory = f"Multiyear/{file_identifier}"
    sorted_multiyear_weights = pd.read_csv(base_directory + f'/sloped_combined_weights_{degree_fit}')

    # Get the top 10 and bottom 10
    top_10 = sorted_multiyear_weights[:10][::-1]
    bottom_10 = sorted_multiyear_weights[-11:]

    # Plot each individually
    for weights, title in [(top_10, "Top"), (bottom_10, "Bottom")]:
        # Extract keys and values from the dictionary
        x = list(weights['Token'])
        y = list(weights['Slope'])

        # Plot the data
        plt.bar(x, y)

        # Set labels and title
        plt.xlabel('Tokens', size=10)
        plt.ylabel('Slopes')
        plt.title(title + f" 10 weights vs. slope, degree {degree_fit}")

        # Rotate x-axis labels if needed
        plt.xticks(rotation=30)

        # Save the fig
        plt.savefig(f'{base_directory}/{title}_{degree_fit}')
        plt.clf()


# Plots the words individually
def plot_individually(file_identifier, degree_fit, words):

    # Get the dataframe
    base_directory = f"Multiyear/{file_identifier}"
    sorted_multiyear_weights = pd.read_csv(base_directory + f'/sloped_combined_weights_{degree_fit}')

    # Define the words
    if len(words) == 0:
        words = list(sorted_multiyear_weights[:2][::-1]['Token']) + list(sorted_multiyear_weights[-3:]['Token'])

    # Filter rows based on 'Token' column
    filtered_df = sorted_multiyear_weights[sorted_multiyear_weights['Token'].str.contains('|'.join(words))]

    # Get the year values
    year_columns = [column_name for column_name in filtered_df.columns if column_name.startswith('Weights_')]
    years = [column_name[-4:] for column_name in year_columns]

    # Extract columns for x-axis and y-axis
    for word in words:
        row = filtered_df.loc[filtered_df['Token'] == word]
        weights = row[year_columns]
        plt.scatter(years, weights, label=word)
        plt.plot(years, weights.iloc[0], color='black')

    # Plot the data with different colors for each word
    plt.xlabel('Year')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.title('Weights for Selected Tokens')

    # Show
    plt.show()


# I put everything in here because it's not worth it to make a new method
if __name__ == '__main__':
    keyword = 'porn'
    augment_predictions = True
    fifty_fifty = False  # if fifty_fifty, it shouldn't be one's accuracy
    ones_accuracy = True  # If ones_accuracy, there shouldn't be a second keyword
    second_keyword = None
    lambda_value = 1e-05
    minimum_appearances_prevalence = 3  # Should always be prevalence, not minimum appearances
    default_amount = 0.02
    max_training_size = 200000

    # The actual analysis
    # run_multiyear_analysis(keyword, augment_predictions, fifty_fifty, ones_accuracy, second_keyword, lambda_value, minimum_appearances_prevalence, default_amount, max_training_size)

    # Process the data
    file_identifier = "nsfw_porn_1prevalence_1e-05lambda_augment"
    combine_multiyear_weights(file_identifier, start_year=-1, normalize=True)

    # Analyze the data
    degree_fit = 1
    analyze_multiyear_weights(file_identifier, degree_fit)

    # Plot the top weights
    plot_top_and_bottom_weights(file_identifier, degree_fit)
    # plot_individually(file_identifier, degree_fit, words=[])
