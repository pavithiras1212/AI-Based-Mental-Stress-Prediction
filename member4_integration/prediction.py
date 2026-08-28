import os
import joblib
import pandas as pd

# Project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model files
MODEL_PATH = os.path.join(BASE_DIR, "best_stress_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")


# Load trained model
model = joblib.load(MODEL_PATH)

# Load feature names used during training
model_features = joblib.load(FEATURES_PATH)


def predict_stress(student_data):
    """
    Predict stress level for one student.

    student_data should be a dictionary containing
    the student's survey responses.
    """

    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame([student_data])

    # Convert categorical responses into numerical features
    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )

    # Make sure input has exactly the same features
    # that were used while training the model
    input_df = input_df.reindex(
        columns=model_features,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_df)

    return prediction[0]


if __name__ == "__main__":

    # Sample student input for testing
    sample_student = {
        "Age": "18–20",
        "Year of Study": "1st Year",
        "How often do you feel stressed?": "Often",
        "What is the main cause of your stress?": "Academic pressure",
        "How many hours do you sleep on average?": "5-6 hours",
        "How often do you exercise or do physical activity?": "Rarely",
        "Have you experienced any of the following due to stress?": "Headache",
        "How do you usually cope with stress?": "Talk with friends"
    }

    result = predict_stress(sample_student)

    print("Predicted Stress Level:", result)