# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('hiring.csv')

# Fill missing values
dataset['experience'] = dataset['experience'].fillna(0)
dataset['test_score'] = dataset['test_score'].fillna(dataset['test_score'].mean())

# Separate features (X) and target variable (y)
X = dataset.iloc[:, :3].copy()
y = dataset.iloc[:, -1].copy()

# Converting words to integer values
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 
                 'twelve': 12, 'zero': 0, 0: 0}
    return word_dict.get(word, 0)  # default to 0 if word is not in dictionary

X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

# Check for any remaining NaNs and fill or drop them
if X.isnull().values.any():
    print("Found NaNs in X. Filling with column means.")
    X.fillna(X.mean(), inplace=True)

# Train the model
regressor = LinearRegression()
regressor.fit(X, y)

# Save the model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Load the model and test a prediction
# Load the model and test a prediction
model = pickle.load(open('model.pkl', 'rb'))

# Define the input data with column names
input_data = pd.DataFrame([[2, 9, 6]], columns=['experience', 'test_score', 'interview_score'])

# Make a prediction
print(model.predict(input_data))