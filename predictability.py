import numpy as np
import pandas as pd
import main
import json
import random
import os
import concurrent.futures
from functools import partial

def find_words_with_similar_prevalence(keyword, year, leeway=0, max_tokens=-1, top_tokens=-1):
    # You shouldn't run this code with top tokens and max_tokens, doesn't really make sense
    assert not (max_tokens != -1 and top_tokens != -1)
    # Same with leeway and top_tokens
    assert not (leeway != 0 and top_tokens != -1)

    # Get the prevalences from that year
    prevalences = pd.read_csv("Datasets/jjjitv2.csv")
    filtered_prevalences = prevalences[prevalences["Year"] == year]
    filtered_prevalences = filtered_prevalences[~filtered_prevalences["Token"].str.contains(' ')]

    # Get the tokens with a similar prevalence as needed
    similar_prevalence_tokens = set()
    if top_tokens == -1:
        # Get the words with the same prevalence +- leeway as a percent, assumes it's in the filtered prevalences
        keyword_prevalence = int(filtered_prevalences[filtered_prevalences["Token"] == keyword]["Prevalence"])
        print(f'The keyword prevalence is {keyword_prevalence}')
        buffer_amount = int(np.ceil(keyword_prevalence * leeway) )
        min_prevalence = keyword_prevalence - buffer_amount
        max_prevalence = keyword_prevalence + buffer_amount
        print(f'The min prevalence to be considered is {min_prevalence}, the max is {max_prevalence}')

        # Get all the words with the same prevalence as the original word
        for prevalence in range(min_prevalence, max_prevalence + 1):
            print(prevalence)
            print(filtered_prevalences[filtered_prevalences['Prevalence'] == prevalence]['Token'])
            similar_prevalence_tokens.update(
                filtered_prevalences[filtered_prevalences['Prevalence'] == prevalence]['Token'])

        # If there is a max and there are more than the max, sample max randomly
        if max_tokens != -1 and len(similar_prevalence_tokens) > max_tokens:
            similar_prevalence_tokens = random.sample(list(similar_prevalence_tokens), max_tokens)
            if keyword not in similar_prevalence_tokens:
                similar_prevalence_tokens.pop()
                similar_prevalence_tokens.append(keyword)
    else:
        sorted_df = filtered_prevalences.sort_values(by='Prevalence', ascending=False)
        selected_rows = sorted_df.head(top_tokens)
        similar_prevalence_tokens = set(selected_rows['Token'])

    # Sanity check
    return similar_prevalence_tokens


# Helper function for parallelizing
def execute_main(token, year, lambda_value, minimum_appearances_prevalence, default_amount, max_training_size, file_prefix, file_path):
    # Get the file that it would be stored in
    file_keyword = token.replace('/', 'BACKSLASH')
    token_file_path = f'{file_keyword}_' \
                      f'{minimum_appearances_prevalence}prevalence_' \
                      f'{lambda_value}lambda_' \
                      f'augment_' \
                      f'onesaccuracy' \
                      f'{"_maxtrainingsize" + str(max_training_size) if max_training_size != -1 else ""}'
    directory_path = os.path.join(file_prefix, token_file_path)
    print(f"Checking if the file already exists at {directory_path}")
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"Token exists at {directory_path} \n")
        # There definitely should be a JSON file called relevant_data_json
        # If there isn't, throw an error I'm not going to try to catch it
        token_file_path = os.path.join(directory_path, 'relevant_data_json')
        with open(token_file_path, 'r') as file:
            token_data = json.load(file)
        accuracies.append((token, (token_data[1], token_data[0])))
    else:
        accuracies.append((token, main.main(file_path,
                          keyword=token,
                          augment_predictions=True,
                          fifty_fifty=False,
                          ones_accuracy=True,
                          second_keyword=None,
                          lambda_value=lambda_value,
                          minimum_appearances_prevalence=minimum_appearances_prevalence,
                          multiyear=False,
                          save_results=True,
                          default_amount=default_amount,
                          max_training_size=max_training_size,
                          prefix_file_path=file_prefix) ) )


# Main code to run
if __name__ == '__main__':
    keyword = 'nsfw'
    lambda_value = 10 ** (-5)
    minimum_appearances_prevalence = 3
    default_amount = 0.02
    max_training_size = 200000
    year = 2022

    # Find words with similar prevalences given a year
    top_tokens = 5
    leeway = 0  # Default 0
    max_tokens = -1  # Default -1
    similar_prevalence_tokens = find_words_with_similar_prevalence(keyword, year, leeway=leeway, max_tokens=max_tokens, top_tokens=top_tokens)
    print(f'There are {len(similar_prevalence_tokens)} tokens with similar prevalence')
    print(similar_prevalence_tokens, '\n')

    # Get the file info
    file_prefix = "All_Tokens" if top_tokens != -1 else "Miscellaneous"
    file_keyword = keyword.replace('/', 'BACKSLASH')
    file_path = f'{file_keyword if top_tokens == -1 else "top" + str(top_tokens)}_' \
                f'{year}year_' \
                f'{lambda_value}lambda_' \
                f'{minimum_appearances_prevalence}minprevalence_' \
                f'{str(default_amount).replace(".", "_")}defaultamount_' \
                f'{max_training_size}maxtrainingsize'
    print(f'Final data will be saved in {os.path.join("Predictability", file_path)} \n')

    # See their test accuracy with the default hyperparameters
    print('Calculating the accuracies for all tokens.')
    accuracies = []
    unhandled_tokens = set()
    parallelize = True
    if parallelize:
        # THIS PARALLELIZE DOESN'T INCLUDE YEAR, it uses the alternative file_path system
        print('Reading the dataset before paralellizing.')
        main_dataframe = pd.read_csv(f'Datasets/one_bio_per_year/one_bio_per_year_{year}.csv', header=None)
        column_names = ['user_id_str', 'bio', 'location', 'name', 'screen_name', 'protected',
                        'verified', 'followers_count', 'friends_count', 'statuses_count', 'created_at',
                        'default_profile', 'us_day_YYYY_MM_DD', 'timestamp_ms']
        main_dataframe.columns = column_names
        bios = main_dataframe['bio'].dropna()  # Drop nan
        print(f'There are {len(bios)} total bios')
        joint_file_path = (bios, 2022)

        # Create a ThreadPoolExecutor with the desired number of threads (28 in our case)
        with concurrent.futures.ThreadPoolExecutor(max_workers=28) as executor:
            # Define the function to be executed in parallel using the executor.map method
            process_tokens = partial(execute_main, year=year, lambda_value=lambda_value,
                                     minimum_appearances_prevalence=minimum_appearances_prevalence,
                                     default_amount=default_amount, max_training_size=max_training_size,
                                     file_prefix=file_prefix, file_path=joint_file_path)

            # Submit the tasks to the executor and retrieve the results
            results = executor.map(process_tokens, similar_prevalence_tokens)

            # Process the results as they become available
            for token, result in zip(similar_prevalence_tokens, results):
                if result is not None:
                    accuracies.append((token, result))
                else:
                    unhandled_tokens.add(token)
    else:
        for token in similar_prevalence_tokens:
            # Get the file that it would be stored in
            token_file_path = f'{token}_' \
                              f'{minimum_appearances_prevalence}prevalence_' \
                              f'{lambda_value}lambda_' \
                              f'augment_' \
                              f'onesaccuracy' \
                              f'{"_maxtrainingsize" + str(max_training_size) if max_training_size != -1 else ""}'
            directory_path = os.path.join(file_prefix, token_file_path)
            print(f"Checking if the file already exists at {directory_path}")
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                print(f"Token exists at {directory_path} \n")
                # There definitely should be a JSON file called relevant_data_json
                # If there isn't, throw an error I'm not going to try to catch it
                token_file_path = os.path.join(directory_path, 'relevant_data_json')
                with open(token_file_path, 'r') as file:
                    token_data = json.load(file)
                accuracies.append( (token, (token_data[1], token_data[0]) ) )
            else:
                print(f'Calculating the accuracies for {token}')
                try:
                    accuracies.append( (token,
                                       main.main(f'Datasets/one_bio_per_year/one_bio_per_year_{year}.csv',
                                                 keyword=token,
                                                 augment_predictions=True,
                                                 fifty_fifty=False,
                                                 ones_accuracy=True,
                                                 second_keyword=None,
                                                 lambda_value=lambda_value,
                                                 minimum_appearances_prevalence=minimum_appearances_prevalence,
                                                 multiyear=False,
                                                 save_results=True,
                                                 default_amount=default_amount,
                                                 max_training_size=max_training_size,
                                                 prefix_file_path=file_prefix ) ) )
                except:
                    unhandled_tokens.add(token)
    # Sort them by their test accuracies
    print(f'The accuracies are {accuracies}')
    sorted_accuracies = sorted(accuracies, key=lambda x: x[1][0], reverse=True)
    # Save them in a file, open the file in write mode
    # Create the directory if it doesn't exist
    os.makedirs(f'Predictability/{file_path}', exist_ok=True)
    with open(f'Predictability/{file_path}/results', 'w') as file:
        # Write the list to the file in JSON format
        json.dump(sorted_accuracies, file)
    # Save the unhandled tokens
    with open(f'Predictability/{file_path}/unhandled_tokens', 'w') as file:
        json.dump(list(unhandled_tokens), file)

    # Give a percentile for the predictability of a word
    # Find the word in the list, this only applies if there were not doing top counts
    if top_tokens == -1:
        keyword_index = -1
        for index, entry in enumerate(sorted_accuracies[::-1]):
            if entry[0] == keyword:
                keyword_index = index
                break
        if keyword_index == -1:
            information_string = "Something broke, go back and fix it!"
        else:
            # The percentile is the keyword_index over the length of the list
            information_string = f'The predictability percentile of {keyword} is {keyword_index / len(sorted_accuracies)}'
            print(f'The predictability percentile of {keyword} is {keyword_index / len(sorted_accuracies)}')
    else:
        information_string = f"There are {len([name for name in os.listdir('All_Tokens')])} tokens whose weights have been calculated"
    # Save the information string
    with open(f'Predictability/{file_path}/information_string', 'w') as file:
        json.dump(information_string, file)
