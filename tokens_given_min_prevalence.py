import pandas as pd

# Set your parameters
year = 2022
min_prevalence = 7

# Get the prevalences from that year
prevalences = pd.read_csv("Datasets/jjjitv2.csv")
filtered_prevalences = prevalences[prevalences["Year"] == year]
filtered_prevalences = filtered_prevalences[~filtered_prevalences["Token"].str.contains(' ')]

# Get all the tokens with at least the min prevalence
min_prevalences = filtered_prevalences[filtered_prevalences['Prevalence'] >= min_prevalence]

# Print data about them
print(f'There are {len(min_prevalences)} tokens that have that minimum prevalence.')