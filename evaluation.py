import json
import numpy as np


# Returns the top n weights
def find_top_n_weights(token_lookup, W, n, reverse):
    sorted_indices = np.argsort(W)
    if not reverse:
        top_indices = sorted_indices[-n:]
    else:
        top_indices = sorted_indices[:n]
    top_weights = {}
    for index in top_indices:
        top_weights[token_lookup[str(index)]] = W[index]
    return top_weights


# Enter analysis mode, where you can search for specific tokens
def enter_analysis_mode():
    print()
    while True:
        keyword = input("Please enter the keyword of interest, or EXIT to exit: ")
        if keyword == "EXIT":
            break
        # This is terrible code, fix this later
        for key, value in token_lookup.items():
            if value == keyword:
                weight = W[int(key)]
                print(f'The weight of {keyword} is {weight}')
                break


# Main method
def main():
    # Globalize important variables
    global token_lookup
    global W

    # Import the objects we need
    with open(f'Results/{file_identifier}/token_lookup', 'r') as file_object:
        token_lookup = json.load(file_object)
    W = np.loadtxt(f'Results/{file_identifier}/W', delimiter=',')

    # Find the top and bottom weights
    top_ten_weights = find_top_n_weights(token_lookup, W, n=10, reverse=False)
    bottom_ten_weights = find_top_n_weights(token_lookup, W, n=10, reverse=True)
    print('The top ten weights are as follows\n', str(top_ten_weights) )
    print('The bottom ten weights are as follows\n', str(bottom_ten_weights) )

    # Find the bias
    print(f'The weights of the bias is {W[len(W) - 1]}')

    # Enter analysis mode
    enter_analysis_mode()


if __name__ == '__main__':
    # TODO fix this later
    global file_identifier
    file_identifier = 'mom_husband_1prevalence_1e-05lambda_augment'

    # TODO make it so that you can search the weight for a specific word compared to the average

    main()