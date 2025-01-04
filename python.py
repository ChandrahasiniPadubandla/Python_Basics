import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Class for processing data related to age and hours of study
class StudentData:
    def __init__(self, age, hours_studied):
        self.age = age
        self.hours_studied = hours_studied

    def to_dataframe(self):
        """Convert student data to a pandas DataFrame."""
        return pd.DataFrame({'Age': self.age, 'Hours_Studied': self.hours_studied})

# Class for handling the prediction of marks
class MarksPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False

    def train(self, data, target):
        """Train the model on the given dataset."""
        self.model.fit(data, target)
        self.trained = True

    def predict(self, new_data):
        """Predict marks based on new student data."""
        if not self.trained:
            raise ValueError("Model is not trained. Train the model before predicting.")
        return self.model.predict(new_data)

    def evaluate(self, data, target):
        """Evaluate the model's performance using Mean Squared Error."""
        if not self.trained:
            raise ValueError("Model is not trained. Train the model before evaluating.")
        predictions = self.model.predict(data)
        mse = mean_squared_error(target, predictions)
        return mse

# Main program
if __name__ == "__main__":
    # Sample dataset
    age = [15, 16, 15, 17, 16, 15, 17, 16, 18, 15]
    hours_studied = [2, 3, 1, 4, 5, 2, 6, 3, 7, 1]
    marks = [60, 70, 55, 80, 85, 65, 90, 75, 95, 50]

    # Create instances of the classes
    student_data = StudentData(age, hours_studied)
    marks_predictor = MarksPredictor()

    # Prepare the data
    df = student_data.to_dataframe()
    X = df
    y = marks

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    marks_predictor.train(X_train, y_train)

    # Evaluate the model
    mse = marks_predictor.evaluate(X_test, y_test)
    print(f"Mean Squared Error: {mse}")

    # Predict new marks
    new_student_data = StudentData(age=[16, 17], hours_studied=[4, 5])
    new_data_df = new_student_data.to_dataframe()
    predictions = marks_predictor.predict(new_data_df)

    print(f"Predicted Marks:\n{predictions}")
