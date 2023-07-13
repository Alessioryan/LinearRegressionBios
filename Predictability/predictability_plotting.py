import json
import matplotlib.pyplot as plt

# Read the JSON file
with open('nsfw_1e-05lambda_3minprevalence_0.02defaultamount_200000maxtrainingsize') as f:
    data = json.load(f)
    # Reverse the data
    data = data[::-1]

# Extract token and test_accuracy from the data
tokens = []
test_accuracies = []
for token, (test_accuracy, train_accuracy) in data:
    tokens.append(token)
    test_accuracies.append(test_accuracy)

# Plot the data
plt.plot(tokens, test_accuracies, 'o-')
plt.xlabel('Token')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Token')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)

# Add a horizontal line at a specific value
line_value = 0.41064120054  # Adjust this value as per your requirement
plt.axhline(line_value, color='red', linestyle='--', label='Threshold')

plt.show()
