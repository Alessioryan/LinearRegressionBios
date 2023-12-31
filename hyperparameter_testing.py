import main
import numpy as np
import json


# Define hyperparameter tuning function
def hyperparameter_tune(lower_lambda, upper_lambda, steps_lambda,
                        lower_appear, upper_appear, steps_appear,
                        lower_default, upper_default, steps_default):
    # Define parameters
    keyword_list = ['porn', 'nsfw']
    augment_predictions = True
    ones_accuracy = True  # If ones_accuracy, there shouldn't be a second keyword
    fifty_fifty = False
    second_keyword = None

    # Define the accuracy lists and other necessary lists
    lambdas = np.logspace(np.log10(lower_lambda), np.log10(upper_lambda), steps_lambda)
    appears = np.linspace(lower_appear, upper_appear, steps_appear, dtype=int)
    defaults = np.linspace(lower_default, upper_default, steps_default)


    # Do this for every item in the keyword list
    for keyword in keyword_list:
        # Save the results to a file
        # Define the unique file identifier
        file_identifier = f'Hyperparameter tunings/{keyword}' \
                          f'_{second_keyword + "_" if second_keyword else ""}' \
                          f'{"_augment" if augment_predictions else ""}' \
                          f'{"_onesaccuracy" if ones_accuracy else ""}' \
                          f'_{lower_lambda, upper_lambda, steps_lambda, lower_appear, upper_appear, steps_appear, lower_default, upper_default, steps_default}.json'
        print(f'Printing to {file_identifier}')

        test_train_accuracies = []
        # Find the optimal hyperparameters
        for lambda_value in lambdas:
            for appears_value in appears:
                for default_value in defaults:
                    print(f'Working with lambda={lambda_value}, appears={appears_value}, default={default_value}')
                    test_train_accuracies.append((lambda_value,
                                                 appears_value,
                                                 default_value,
                                                 main.main('Datasets/one_bio_per_year/one_bio_per_year_2022.csv',
                                                           keyword=keyword,
                                                           augment_predictions=augment_predictions,
                                                           fifty_fifty=fifty_fifty,
                                                           ones_accuracy=ones_accuracy,
                                                           second_keyword=second_keyword,
                                                           lambda_value=lambda_value,
                                                           minimum_appearances_prevalence=appears_value,
                                                           save_results="none",
                                                           default_amount=default_value) ) )
                    print(test_train_accuracies[-1])

        # Sort the file
        converted_tuples = [(x[0], int(x[1]), x[2], x[3]) for x in test_train_accuracies]
        sorted_list = sorted(converted_tuples, key=lambda x: x[3][0], reverse=True)
        print("The sorted list is ", sorted_list)

        # Write the list to the JSON file
        with open(file_identifier, "w") as file:
            json.dump(sorted_list, file)


# Run the code!!
if __name__ == '__main__':
    hyperparameter_tune(10**(-5), 10**(-5), 1,
                        3, 3, 1,
                        0.02, 0.1, 6)