import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Plots the accuracies against the selected hyperparameter
    # 0 for lambda, 1 for appearances, 2 for default value
    hyperparameter = 1

    # File name here
    file_name = 'porn__augment_onesaccuracy_(1e-05, 1e-05, 1, 1, 6, 6, 0.1, 0.1, 1).json'
    file_path = file_name

    # Read JSON file
    with open(file_path) as file:
        data = json.load(file)

    # Extract values from the JSON data
    x_axis = [entry[hyperparameter] for entry in data]
    y_test = [entry[3][0] for entry in data]
    y_train = [entry[3][1] for entry in data]

    # Rename hyperparameter for title
    if hyperparameter == 0:
        hyperparameter = 'Lambda'
    elif hyperparameter == 1:
        hyperparameter = 'Min appearances'
    elif hyperparameter == 2:
        hyperparameter = 'Default value'
    else:
        raise ValueError('Please enter 0, 1, or 2 as the hyperparameter')

    # Plot the data and best fit lines
    plt.scatter(x_axis, y_test, label='test accuracy')
    plt.scatter(x_axis, y_train, label='train accuracy')
    plt.xlabel(hyperparameter)
    plt.ylabel('Test and train accuracy')
    plt.legend()
    if hyperparameter == 'Lambda':
        plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.savefig(file_path.replace('json', 'png') )