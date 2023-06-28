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
    return bio_tokens


# Construct a vocabulary. A token is in the vocabulary if it appears in at least n bios
def construct_vocabulary(keyword, second_keyword, minimum_appearances, is_prevalence):
    raw_vocabulary = defaultdict(int)
    for bio in bios:
        for token in set(tokenize(bio) ):
            raw_vocabulary[token] += 1

    # We don't want to look at our keywords in the vocabulary
    del raw_vocabulary[keyword]
    if second_keyword:
        del raw_vocabulary[second_keyword]

    # Remove any token that appears less than the minimum_appearances threshold
    vocabulary = defaultdict(int)
    for key, value in raw_vocabulary.items():
        if is_prevalence:
            if value / len(bios) * 10000 > minimum_appearances:
                vocabulary[key] = value
        else:
            if value > minimum_appearances:
                vocabulary[key] = value
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
def predict(keyword, second_keyword, W, X, Y, augment_predictions):
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


def main(file_path, keyword, augment_predictions, ones_accuracy, second_keyword, lambda_value, minimum_appearances_prevalence, save_results=True):
    # TODO fix this later
    # Define global variables
    global bios
    global ordered_vocabulary
    global expected_percent

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
    bios = data_frame['bio']
    print(f'There are {len(bios)} total bios')

    # If there is a second keyword, filter the data preemptively
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
    print(f'\n       There are {len(bios)} relevant bios')

    # Construct a vocabulary
    is_prevalence = True
    vocabulary = sorted(construct_vocabulary(keyword, second_keyword, minimum_appearances_prevalence, is_prevalence) )
    ordered_vocabulary = {}
    token_lookup = {}
    for index, token in enumerate(vocabulary):
        ordered_vocabulary[token] = index
        token_lookup[index] = token
    # Order tokens alphabetically

    # Create train/test split, 90% train, 10% test
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
    file_identifier = f'{keyword}_' \
                      f'{second_keyword + "_" if second_keyword else ""}' \
                      f'{minimum_appearances_prevalence}' \
                      f'{"prevalence" if is_prevalence else "appearances"}_' \
                      f'{lambda_value}lambda' \
                      f'{"_augment" if augment_predictions else ""}' \
                      f'{"_onesaccuracy" if ones_accuracy else ""}'

    # Calculate W
    print('\nCalculating W...')
    W = np.linalg.solve(np.matmul(X.T, X) + lambda_I, np.matmul(X.T, Y) )
    if save_results:
        np.savetxt(f'Weights/{file_identifier}', W, delimiter=',', fmt='%f')

    # Evaluate the accuracy of W on the test and train sets
    print('Calculating the predictions on the training set')
    preds_train, raw_scores, threshold = predict(keyword, second_keyword, W, X, Y, augment_predictions)

    # Save information in results
    if save_results:
        os.makedirs(f'Results/{file_identifier}', exist_ok=True)
        np.savetxt(f'Results/{file_identifier}/W', W, delimiter=',', fmt='%f')
        np.savetxt(f'Results/{file_identifier}/Y', Y, delimiter=',', fmt='%d')
        np.savetxt(f'Results/{file_identifier}/preds', preds_train, delimiter=',', fmt='%d')
        token_lookup_file_path = f'Results/{file_identifier}/token_lookup'
        with open(token_lookup_file_path, 'w') as file:
            json.dump(token_lookup, file)
    # Finds the training accuracy
    accuracy = find_accuracy(ones_accuracy, preds_train, train_bios, Y)
    train_accuracy_data = f'The train accuracy is {accuracy}\n'
    print(train_accuracy_data)

    # Find the test accuracy
    print('Calculating the predictions on the testing set')
    X_test = np.zeros((test_bios.shape[0], len(vocabulary)))
    Y_test = np.zeros((test_bios.shape[0],))
    X_test, Y_test = fill_values(keyword, second_keyword, test_bios, X_test, Y_test)
    preds_test, _, _ = predict(keyword, second_keyword, W, X_test, Y_test, augment_predictions)
    accuracy = find_accuracy(ones_accuracy, preds_test, test_bios, Y_test)
    test_accuracy_data = f'The test accuracy is {accuracy}\n'
    print(test_accuracy_data)

    # Save the predictions and the true values
    if save_results:
        Y_preds_raw_bios = pd.DataFrame({
            'Y': Y,
            'preds_train': preds_train,
            'raw_scores': raw_scores,
            'bios': train_bios
        })
        Y_preds_raw_bios.to_csv(f'Results/{file_identifier}/Y_and_preds')

        # Save the non-zero ones separately
        nonzero_Y_preds_raw_bios = Y_preds_raw_bios[(Y_preds_raw_bios['Y'] + Y_preds_raw_bios['preds_train']).values != 0.0]
        nonzero_Y_preds_raw_bios.to_csv(f'Results/{file_identifier}/nonzero_Y_and_preds')

        # Save other relevant data
        threshold_data = f'The threshold for being selected is a score of {threshold}'
        relevant_data = [train_accuracy_data, test_accuracy_data, threshold_data]
        with open(f'Results/{file_identifier}/relevant_data', "w") as file:
            file.writelines("\n".join(relevant_data))

        # Save the weights with their definitions
        weights_with_tokens = pd.DataFrame({
            'Weights': W,
            'Token': token_lookup.items()
        })
        weights_with_tokens.to_csv(f'Results/{file_identifier}/weights_with_tokens', index=False)

        # Sort it
        sorted_weights_with_tokens = weights_with_tokens.sort_values(by='Weights', ascending=False)
        sorted_weights_with_tokens.to_csv(f'Results/{file_identifier}/sorted_weights_with_tokens', index=False)

    # Return the test and train accuracy
    return test_accuracy_data, train_accuracy_data


if __name__ == '__main__':
    keyword = 'porn'
    augment_predictions = True
    ones_accuracy = True  # If ones_accuracy, there shouldn't be a second keyword
    second_keyword = 'nsfw'
    lambda_value = 1e-05
    minimum_appearances_prevalence = 5

    main('Datasets/one_bio_per_year_2022.csv',
         keyword=keyword,
         augment_predictions=augment_predictions,
         ones_accuracy=ones_accuracy,
         second_keyword=second_keyword,
         lambda_value=lambda_value,
         minimum_appearances_prevalence=minimum_appearances_prevalence,
         save_results=True)