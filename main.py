import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os
import json
import re


# Given a bio, returns the tokens in that bio as a list
def tokenize(bio):
    bio = str(bio).casefold()
    bio_tokens = re.split(r"\b|\s+", bio)
    return set(bio_tokens)


# Construct a vocabulary. A token is in the vocabulary if it appears in at least n bios
def construct_vocabulary(keyword, second_keyword, minimum_appearances, is_prevalence, year=None):
    if year is None or not is_prevalence:
        # Tokenize all bios and create a DataFrame
        tokens_df = pd.DataFrame(bios, columns=['bio'])
        tokens_df['tokens'] = tokens_df['bio'].apply(tokenize)

        # Flatten the tokens column into individual rows
        tokens_flat = tokens_df.explode('tokens')

        # Count the occurrences of each token
        raw_vocabulary = tokens_flat['tokens'].value_counts().to_dict()

        # Create a DataFrame from the raw_vocabulary dictionary
        vocab_df = pd.DataFrame.from_dict(raw_vocabulary, orient='index', columns=['count'])

        # Filter the vocabulary based on the minimum_appearances threshold
        if is_prevalence:
            vocab_df = vocab_df[vocab_df['count'] / len(bios) * 10000 > minimum_appearances]
        else:
            vocab_df = vocab_df[vocab_df['count'] > minimum_appearances]

        # There's gotta be a better way to write this
        vocabulary = set(vocab_df.index)
    else:
        # Load in the prevalences for that year
        year_prevalence = pd.read_csv("Datasets/jjjitv2.csv")
        year_prevalence = year_prevalence[year_prevalence["Year"] == int(year)]

        # Filter ones with too low of a prevalence
        year_prevalence = year_prevalence[year_prevalence["Prevalence"] >= minimum_appearances]

        # TODO add a ngram option
        # Remove all ngrams
        year_prevalence = year_prevalence[~year_prevalence['Token'].str.contains(' ')]

        # Create a vocabulary out of this
        vocabulary = set(year_prevalence["Token"])

    # Remove the keywords from the vocabulary
    if keyword in vocabulary:
        vocabulary.remove(keyword)
    if second_keyword and second_keyword in vocabulary:
        vocabulary.remove(second_keyword)

    print(f'The size of the vocabulary is {len(vocabulary)}')
    return vocabulary


# Given the X and the Y and the token of interest, fill in X and Y
def fill_values(keyword, second_keyword, train_test_bios, X, Y):
    # Fill with the values
    for index, bio in enumerate(train_test_bios):
        for token in set(tokenize(bio) ):
            if token == keyword:
                Y[index] = 1
            elif second_keyword and token == second_keyword:
                Y[index] = -1
            elif token in ordered_vocabulary.keys():  # TODO Allow for OOV options
                X[index, ordered_vocabulary[token]] += 1
    # Sanity check for two keywords
    if second_keyword:
        assert np.all(Y != 0)
    # Add the bias
    for index in range(len(X) ):
        X[index, -1] = 1
    return X, Y


# Creates a prediction matrix
def predict(second_keyword, W, X, Y, augment_predictions):
    predictions = np.zeros(shape=(Y.shape[0], ) )
    # Set the default value to -1 if there's a second word
    if second_keyword:
        predictions -= 1

    # Calculate the raw score
    raw_scores = np.matmul(W.T, X.T)
    if augment_predictions:
        # TODO this is DEFINITELY not the best way to calculate this, I'm sinning to the ML gods, but it works for now
        expected_num_winners = int(expected_percent * len(Y) )
        top_indices = np.argsort(raw_scores)[-expected_num_winners:]
        threshold = raw_scores[top_indices[1]]
        print(f'The threshold for being selected is a score of {threshold}')
        for index in top_indices:
            predictions[index] = 1
    else:
        predictions = np.where(raw_scores >= (0 if not second_keyword else 1), 1, -1)
        threshold = None

    # Return the beautiful result
    return predictions, raw_scores, threshold


# If ones_accuracy, calculates the accuracy only on the ones contining a one,
# otherwise calculates regular accuracy, which is the difference between the preds and the Y
def find_accuracy(ones_accuracy, preds, bios, Y):
    if ones_accuracy:
        non_zero_indices = np.nonzero((preds != 0) | (Y != 0))
        non_zero_preds = preds[non_zero_indices]
        non_zero_Y = Y[non_zero_indices]
        accuracy = 1 - (np.sum(np.abs(non_zero_preds - non_zero_Y)) / non_zero_preds.shape[0])
    else:
        accuracy = np.mean(preds - Y == 0)
    return accuracy


# Returns a tuple (test_accuracy, train_accuracy)
def main(file_path, keyword, augment_predictions, fifty_fifty, ones_accuracy, second_keyword, lambda_value,
         minimum_appearances_prevalence, multiyear=False, save_results=True, default_amount=None,
         max_training_size=-1, prefix_file_path=""):

    # TODO fix this later
    # Define global variables
    global bios
    global ordered_vocabulary
    global expected_percent

    # Sanity checks
    # Can't have second keyword and one's accuracy
    assert not (second_keyword is not None and ones_accuracy)
    # fifty_fifty and default_amount overlap
    assert not (fifty_fifty and default_amount is not None)
    # Can't have a second keyword and default_amount
    assert not (default_amount is not None and second_keyword is not None)
    # Default amount must between 0 and 1
    if default_amount is not None:
        assert not ( (default_amount > 1) or (default_amount < 0) )

    # If you pass in a string, behave as normal:
    if isinstance(file_path, str):
        # Find the year and state that you're working on that year
        year = file_path[-8:-4]
        if multiyear:
            print(f'Working with year {year}')

        # Get the data
        print('Reading the input file')
        if 'sampled' not in file_path:
            data_frame = pd.read_csv(file_path, header=None)
            column_names = ['user_id_str', 'bio', 'location', 'name', 'screen_name', 'protected',
                            'verified', 'followers_count', 'friends_count', 'statuses_count', 'created_at',
                            'default_profile', 'us_day_YYYY_MM_DD', 'timestamp_ms']
            data_frame.columns = column_names
        else:
            data_frame = pd.read_csv(file_path)
        bios = data_frame['bio'].dropna()  # Drop nan
        print(f'There are {len(bios)} total bios')
    else:
        print(f'Dataframe already supplied for keyword {keyword}')
        # It's gotta be a tuple with the dataframe and the year
        bios = file_path[0]
        year = file_path[1]

    # If there is a default amount, then filter some of the data
    if default_amount is not None:
        print(f'Filtering by keyword for {keyword}')
        # Get the bios with and without the keyword
        keyword_regex = rf"\b{keyword}\b"  # TODO does this capture the same keywords as tokenization?
        bios_with_keyword = bios[bios.str.contains(keyword_regex, regex=True)]
        bios_without_keyword = bios[~bios.str.contains(keyword_regex, regex=True)]

        print(f'Filtering bios to get to default amount for {keyword}')
        # EMILY - GETS THE 2% OF BIOS WITH THE KEYWORD
        if len(bios_with_keyword) / len(bios) < default_amount:
            total_required_bios = int(len(bios_with_keyword) / default_amount)
            total_without_keyword_bios = total_required_bios - len(bios_with_keyword)

            # Sample the right amount of bios
            sampled_bios_without_keyword = bios_without_keyword.sample(n=total_without_keyword_bios, replace=False, random_state=42)

            # Concatenate the shuffled dataframe
            bios = pd.concat([bios_with_keyword, sampled_bios_without_keyword], ignore_index=True)

            # Shuffle the dataframe
            bios = bios.sample(frac=1, random_state=42)
        elif len(bios_without_keyword) / len(bios) < default_amount:
            total_required_bios = int(len(bios_without_keyword) / default_amount)
            total_with_keyword_bios = total_required_bios - len(bios_without_keyword)

            # Sample the right amount of bios
            sampled_bios_with_keyword = bios_without_keyword.sample(n=total_with_keyword_bios, replace=False, random_state=42)

            # Concatenate the shuffled dataframe
            bios = pd.concat([bios_without_keyword, sampled_bios_with_keyword], ignore_index=True)

            # Shuffle the dataframe
            bios = bios.sample(frac=1, random_state=42)

    # If there is a second keyword, filter the data
    if second_keyword:
        keyword_regex = rf"\b{keyword}\b"
        second_keyword_regex = rf"\b{second_keyword}\b"
        contains_first = (bios.str.contains(keyword_regex, regex=True) )
        contains_second = (bios.str.contains(second_keyword_regex, regex=True) )
        contains_both = (bios.str.contains(keyword_regex, regex=True)
                         & bios.str.contains(second_keyword_regex, regex=True) )
        contains_first_or_second = (contains_first | contains_second) & ~contains_both
        print(f'{np.mean(contains_first)} contain {keyword}, {np.mean(contains_second)} contain {second_keyword}')
        print(f'{np.mean(contains_both)} contain both, giving {np.mean(contains_first_or_second)} containing only one')
        bios = bios[contains_first_or_second]

    # If there is only one keyword and fifty_fifty is on, then get rid of enough bios so that it's actually 50/50
    if fifty_fifty:
        # TODO allow for two keyword fifty_fifty
        # TODO make sure this type of keyword finding covers the same tokenization as tokenize()
        keyword_regex = rf"\b{keyword}\b"
        contains_keyword = bios.str.contains(keyword_regex, regex=True)
        bios_with_keyword = bios[contains_keyword]
        bios_without_keyword = bios[~contains_keyword]
        same_size_no_keyword = bios_without_keyword.sample(n=len(bios_with_keyword), replace=False, random_state=42)
        combined_bios = pd.concat([bios_with_keyword, same_size_no_keyword], ignore_index=True)
        bios = combined_bios.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'\n -----> There are {len(bios)} relevant bios for {keyword}')

    # Construct a vocabulary
    print(f'Building a vocabulary')
    is_prevalence = True
    vocabulary = sorted(construct_vocabulary(keyword, second_keyword, minimum_appearances_prevalence, is_prevalence, year) )
    ordered_vocabulary = {}
    token_lookup = {}
    for index, token in enumerate(vocabulary):
        ordered_vocabulary[token] = index
        token_lookup[index] = token
    # Order tokens alphabetically

    # If we have too many bios, then we must filter some out randomly
    # TODO Make this more precise
    # EMILY - GET THE 200,000
    if max_training_size != -1 and len(bios) > max_training_size:
        bios = bios.sample(n=max_training_size, replace=False, random_state=42)

    # Find the percent of bios that contain the keyword
    percent_contains = np.sum(bios.str.contains(rf"\b{keyword}\b")) / len(bios)
    print(f'The percent of bios with {keyword} is {percent_contains}')

    # Create train/test split, 90% train, 10% test
    # EMILY - 90% 10% split
    train_bios, test_bios = train_test_split(bios, test_size=0.1, random_state=42)

    # Create empty X with bias, create empty Y
    # X is the length of the vocabulary (without keywords) + 1 for the bias
    X = np.zeros((train_bios.shape[0], len(vocabulary) ) )
    Y = np.zeros((train_bios.shape[0], ) )

    # Fill X and Y with values
    X, Y = fill_values(keyword, second_keyword, train_bios, X, Y)
    expected_percent = np.sum(Y == 1) / len(Y)  # TODO Check this works
    print(f'The expected number of bios with {keyword} is {expected_percent}')

    # Define lambda
    lambda_I = lambda_value * np.identity(len(vocabulary))

    # Define the unique file identifier
    file_keyword = keyword.replace('/', 'BACKSLASH')
    file_second_keyword = second_keyword.replace('/', 'BACKSLASH') + "_" if second_keyword else ""
    file_identifier = f'{file_keyword}_' \
                      f'{file_second_keyword}' \
                      f'{"_sampled" if (isinstance(file_path, str) and "sampled" in file_path) else ""}' \
                      f'{minimum_appearances_prevalence}{"prevalence" if is_prevalence else "appearances"}_' \
                      f'{"_fiftyfifty" if fifty_fifty else ""}' \
                      f'{lambda_value}lambda' \
                      f'{"_augment" if augment_predictions else ""}' \
                      f'{"_onesaccuracy" if ones_accuracy else ""}' \
                      f'{"_maxtrainingsize" + str(max_training_size) if max_training_size != -1 else ""}'

    if multiyear:
        file_identifier += f'/{year}'

    # Calculate W
    print('\nCalculating W...')
    W = np.linalg.solve(np.matmul(X.T, X) + lambda_I, np.matmul(X.T, Y) )
    if save_results and not multiyear:
        np.savetxt(f'Weights/{file_identifier}', W, delimiter=',', fmt='%f')

    # Evaluate the accuracy of W on the test and train sets
    print('Calculating the predictions on the training set')
    preds_train, raw_scores, threshold = predict(second_keyword, W, X, Y, augment_predictions)

    # Save information in results
    if prefix_file_path:
        save_directory = prefix_file_path
    elif multiyear:
        save_directory = "Multiyear"
    else:
        save_directory = "Results"
    if save_results:
        os.makedirs(f'{save_directory}/{file_identifier}', exist_ok=True)
        np.savetxt(f'{save_directory}/{file_identifier}/W', W, delimiter=',', fmt='%f')
        np.savetxt(f'{save_directory}/{file_identifier}/Y', Y, delimiter=',', fmt='%d')
        np.savetxt(f'{save_directory}/{file_identifier}/preds', preds_train, delimiter=',', fmt='%d')
        token_lookup_file_path = f'{save_directory}/{file_identifier}/token_lookup'
        with open(token_lookup_file_path, 'w') as file:
            json.dump(token_lookup, file)
    # Finds the training accuracy
    train_accuracy = find_accuracy(ones_accuracy, preds_train, train_bios, Y)
    train_accuracy_data = f'The train accuracy is {train_accuracy}\n'
    print(train_accuracy_data)

    # Find the test accuracy
    print('Calculating the predictions on the testing set')
    X_test = np.zeros((test_bios.shape[0], len(vocabulary)))
    Y_test = np.zeros((test_bios.shape[0],))
    X_test, Y_test = fill_values(keyword, second_keyword, test_bios, X_test, Y_test)
    preds_test, _, _ = predict(second_keyword, W, X_test, Y_test, augment_predictions)
    test_accuracy = find_accuracy(ones_accuracy, preds_test, test_bios, Y_test)
    test_accuracy_data = f'The test accuracy is {test_accuracy}\n'
    print(test_accuracy_data)

    # Save the predictions and the true values
    if save_results:
        Y_preds_raw_bios = pd.DataFrame({
            'Y': Y,
            'preds_train': preds_train,
            'raw_scores': raw_scores,
            'bios': train_bios
        })
        Y_preds_raw_bios.to_csv(f'{save_directory}/{file_identifier}/Y_and_preds')

        # Save the non-zero ones separately
        nonzero_Y_preds_raw_bios = Y_preds_raw_bios[(Y_preds_raw_bios['Y'] + Y_preds_raw_bios['preds_train']).values != 0.0]
        nonzero_Y_preds_raw_bios.to_csv(f'{save_directory}/{file_identifier}/nonzero_Y_and_preds')

        # Save other relevant data
        threshold_data = f'The threshold for being selected is a score of {threshold}'
        relevant_data = [train_accuracy_data, test_accuracy_data, threshold_data]
        with open(f'{save_directory}/{file_identifier}/relevant_data', "w") as file:
            file.writelines("\n".join(relevant_data))
        with open(f'{save_directory}/{file_identifier}/relevant_data_json', "w") as file:
            json.dump([train_accuracy, test_accuracy, threshold], file)

        # Save the weights with their definitions
        weights_with_tokens = pd.DataFrame({
            'Weights': W,
            'Token': token_lookup.values()
        })
        weights_with_tokens.to_csv(f'{save_directory}/{file_identifier}/weights_with_tokens', index=False)

        # Sort it
        sorted_weights_with_tokens = weights_with_tokens.sort_values(by='Weights', ascending=False)
        sorted_weights_with_tokens.to_csv(f'{save_directory}/{file_identifier}/sorted_weights_with_tokens', index=False)

    # Return the test and train accuracy
    return test_accuracy, train_accuracy


if __name__ == '__main__':
    keyword = 'porn'
    augment_predictions = True
    fifty_fifty = False  # if fifty_fifty, it shouldn't be one's accuracy
    ones_accuracy = True  # If ones_accuracy, there shouldn't be a second keyword
    second_keyword = None
    lambda_value = 10**(-5)
    minimum_appearances_prevalence = 3
    multiyear = False
    default_amount = 0.02
    max_training_size = 200000

    main('Datasets/one_bio_per_year/one_bio_per_year_2022.csv',
         keyword=keyword,
         augment_predictions=augment_predictions,
         fifty_fifty=fifty_fifty,
         ones_accuracy=ones_accuracy,
         second_keyword=second_keyword,
         lambda_value=lambda_value,
         minimum_appearances_prevalence=minimum_appearances_prevalence,
         multiyear=multiyear,
         save_results=True,
         default_amount=default_amount,
         max_training_size=max_training_size,
         prefix_file_path="")