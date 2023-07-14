# LinearRegressionBios

Main.main's parameters:
- file_path = the file you want to read. Include "Datasets/" in it, "Datasets/" is in .gitignore
  - Generally, you can pass in a string to this
  - Alternatively, you can pass in a TUPLE
    - The first element is a dataframe
    - The second element is the year of the dataframe
- keyword = the keyword you want to look at
- augment_predictions = if p is the fraction of bios with the keyword, then if augment_predictions = True, the model will predict that there are p% bios with the keyword when evaluating accuracy. If False, then it will predict a score of >0.5 contains the keyword for one keyword and >0 for two keywords
- fifty_fifty = if true, then half the training and test data will contain the keyword. Doesn't work for two keywords yet, and assumes the keyword appears in less than half the bios in the input dataframe
- ones_accuracy = if true, then the accuracy is threat score (TS) aka critical success index (CSI). If false, then it's the true positive. Only makes sense for one keyword
- second_keyword = the second keyword you want to look at. If you only want to look at one keyword, set to None
- lambda_value = a regularizer for linear regression. You may want to find the optimal one in hyperparameter_testing.py
- minimum_appearances_prevalence = the minimum number of appearances/prevalence in the data for a token to count in the vocab. It's prevalence by default, but you can change it to appearances by changing is_prevalence to False
- multiyear = (by default False) formats the output files in the Multiyear folder rather than the Results folder. Used in multiyear.py
- save_results = (by default "all") saves all the results in files, can be set to "none" for no results or "essential" for sorted_weights_with_tokens, relevant_data, and relevant_data_json
- default_amount = (by default None) if None, then runs regression on the whole data file. Otherwise, it's the percentage of the file that we want to look at. Only works for one keyword. Assume the keyword is in p% of bios.
  - Enter number between 0 and 1. Call this n. This represents the percentage of bios with the keyword that we want to run the model on
  - If n < p, then this will select all bios.
  - If n > p, then the model will discard bios without the keyword randomly until n% of bios in the input data contain the keyword
  - if default_amount is not None:
    - make sure fifty_fifty is False
    - make sure second_keyword is None
    - make sure the number is between 0 and 1
- max_training_size = (by default -1) 
  - If -1, then runs on all the data. 
  - If the amount of bios that are available given other parameters is less than max_training size, runs on all
  - Otherwise, runs on max_training_size, randomly selecting from all the filtered bios until we're at that proportion
- prefix_file_path = (by default ""), puts that prefix on all the file paths it saves to


TODOs:
- Check to see if certain semantic groups of keywords are good or bad predictors of a certain word
- Cross year stuff
- Make everything compatible with ngrams
- Replace all string combining instances to build file paths with f'directory/{file_name}' etc. with os.join
  - IF YOU ARE WORKING ON a Windows, system and I haven't done this, the code won't work
  - Please open a Github issue and I'll get to it
  - Also, the naming adds single quotes around keywords that start with "/"