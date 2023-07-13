import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data from file
with open('../Hyperparameter tunings/porn__augment.json', 'r') as file:
    data = json.load(file)

# Extract the third and fourth numbers from the JSON data
third_numbers = [entry[2] for entry in data]
fourth_numbers = [entry[3][0] for entry in data]

# Create the scatter plot
plt.scatter(third_numbers, fourth_numbers)

# Fit a polynomial of degree 2
coefficients = np.polyfit(third_numbers, fourth_numbers, 2)
polynomial = np.poly1d(coefficients)

# Generate points along the x-axis for the polynomial curve
x_values = np.linspace(min(third_numbers), max(third_numbers), 100)

# Calculate corresponding y-values using the polynomial function
y_values = polynomial(x_values)

# Plot the polynomial curve
plt.plot(x_values, y_values, color='red', label='Best Fit Curve (Degree 2)')

# Set labels for the x-axis and y-axis
plt.xlabel('Percent Keyword')
plt.ylabel('Test Accuracy')

# Add a title to the graph
plt.title('Percent Keyword vs. Test Accuracy')

# Display the graph
plt.show()
