import streamlit as st
import sys
import os

# Connect Streamlit with backend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBER4_DIR = os.path.join(BASE_DIR, "member4_integration")

sys.path.insert(0, MEMBER4_DIR)

from prediction import predict_stress
from recommendation_engine import generate_recommendations


st.set_page_config(
    page_title="AI Mental Stress Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI-Based Mental Stress Prediction")

st.write(
    "Enter your details below to predict your stress level "
    "and receive personalized recommendations."
)

st.divider()

st.header("👨‍🎓 Student Details")


# Age
age = st.selectbox(
    "Age",
    [
        "Below 18",
        "18–20",
        "21 - 22",
        "Above 23",
        "30",
        "31"
    ]
)


# Year of Study
year = st.selectbox(
    "Year of Study",
    [
        "1st Year",
        "2nd Year",
        "3rd Year",
        "4th Year",
        "5 th year"
    ]
)


# Stress frequency
stress_frequency = st.selectbox(
    "How often do you feel stressed?",
    [
        "Never",
        "Rarely",
        "Sometimes",
        "Often",
        "Always"
    ]
)


# Main cause
stress_cause = st.selectbox(
    "What is the main cause of your stress?",
    [
        "Academic pressure",
        "Family/Personal issues",
        "Future career",
        "Exams",
        "Other",
        "Financial issues"
    ]
)


# Sleep
sleep = st.selectbox(
    "How many hours do you sleep on average?",
    [
        "Less than 5",
        "5–6",
        "6–8",
        "More than 8"
    ]
)


# Exercise
exercise = st.selectbox(
    "How often do you exercise or do physical activity?",
    [
        "Never",
        "Rarely",
        "Weekly",
        "Daily"
    ]
)


# Stress symptoms
symptoms = st.selectbox(
    "Have you experienced any of the following due to stress?",
    [
        "Headache",
        "Poor sleep",
        "Anxiety, Difficulty concentrating, Poor sleep, Headache",
        "Anxiety, Headache",
        "Anxiety, Poor sleep",
        "Difficulty concentrating",
        "Anxiety",
        "Poor sleep, Headache",
        "Anxiety, Difficulty concentrating, Headache",
        "Difficulty concentrating, Headache",
        "Anxiety, Difficulty concentrating, Poor sleep",
        "Difficulty concentrating, Poor sleep, Headache",
        "Anxiety, Difficulty concentrating"
    ]
)


# Coping
coping = st.text_input(
    "How do you usually cope with stress?",
    placeholder="Example: Talk with friends"
)


st.divider()


# Prediction
if st.button("🔍 Predict Stress", use_container_width=True):

    student = {
        "Age": age,
        "Year of Study": year,
        "How often do you feel stressed?": stress_frequency,
        "What is the main cause of your stress?": stress_cause,
        "How many hours do you sleep on average?": sleep,
        "How often do you exercise or do physical activity?": exercise,
        "Have you experienced any of the following due to stress?": symptoms,
        "How do you usually cope with stress?": coping
    }

    try:

        predicted_stress = predict_stress(student)

        st.success("Prediction completed successfully!")

        st.subheader("📊 Prediction Result")

        st.metric(
            label="Predicted Stress Level",
            value=str(predicted_stress)
        )


        recommendations = generate_recommendations(student)

        st.subheader("💡 Recommended Strategies")

        for recommendation in recommendations:
            st.write("✅ " + recommendation)


    except Exception as e:

        st.error("Something went wrong while making the prediction.")

        st.write("Error:", e)


st.divider()

st.caption(
    "Note: This prediction is based on survey data and statistical "
    "modelling. It is not a medical diagnosis."
)