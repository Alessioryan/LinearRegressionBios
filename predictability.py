import pandas as pd
import main
import json

def find_words_with_similar_prevalence(keyword, year, buffer=0.05):
    # Get the prevalences from that year
    prevalences = pd.read_csv("Datasets/jjjitv2.csv")
    filtered_prevalences = prevalences[prevalences["Year"] == year]
    filtered_prevalences = filtered_prevalences[~filtered_prevalences["Token"].str.contains(' ')]

    # Get the words with the same prevalence +- 5%, assumes it's in the filtered prevalences
    keyword_prevalence = int(filtered_prevalences[filtered_prevalences["Token"] == keyword]["Prevalence"])
    min_prevalence = int(keyword_prevalence * (1 - buffer))
    max_prevalence = int(keyword_prevalence * (1 + buffer))

    # Get all the words with the same prevalence as the original word
    similar_prevalence_tokens = set()
    for prevalence in range(min_prevalence, max_prevalence + 1):
        similar_prevalence_tokens.update(
            filtered_prevalences[filtered_prevalences['Prevalence'] == prevalence]['Token'])

    # Sanity check
    return similar_prevalence_tokens


# Main code to run
if __name__ == '__main__':
    keyword = 'porn'
    lambda_value = 10 ** (-5)
    minimum_appearances_prevalence = 3
    default_amount = 0.02
    max_training_size = 200000
    year = 2022

    # Find words with similar prevalences given a year
    similar_prevalence_tokens = find_words_with_similar_prevalence('nsfw', year)
    print(f'There are {len(similar_prevalence_tokens)} tokens with similar prevalence')
    print(similar_prevalence_tokens, '\n')

    # See their test accuracy with the default hyperparameters
    print('Calculating the accuracies')
    accuracies = []
    for token in similar_prevalence_tokens:
        print(f'Calculating the accuracies for {token}')
        accuracies.append( (token,
                           main.main(f'Datasets/one_bio_per_year/one_bio_per_year_{year}.csv',
                                     keyword=keyword,
                                     augment_predictions=True,
                                     fifty_fifty=False,
                                     ones_accuracy=True,
                                     second_keyword=None,
                                     lambda_value=lambda_value,
                                     minimum_appearances_prevalence=minimum_appearances_prevalence,
                                     multiyear=False,
                                     save_results=False,
                                     default_amount=default_amount,
                                     max_training_size=max_training_size) ) )
    # Sort them by their test accuracies
    sorted_accuracies = sorted(accuracies, key=lambda x: x[1][0], reverse=True)
    # Save them in a file
    file_path = f'Predictability/{keyword}_' \
                f'{lambda_value}lambda_' \
                f'{minimum_appearances_prevalence}minprevalence_' \
                f'{default_amount}defaultamount_' \
                f'{max_training_size}maxtrainingsize'
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the list to the file in JSON format
        json.dump(sorted_accuracies, file)

    # Give a percentile for the predictability of a word
    # Find the word in the list
    keyword_index = -1
    for index, entry in enumerate(sorted_accuracies[::-1]):
        if entry == keyword:
            keyword_index = index
            break
    # Sanity check
    assert keyword_index != -1
    # The percentile is the keyword_index over the length of the list
    print(f'The predictability percentile of {keyword} is {keyword_index / len(sorted_accuracies)}')