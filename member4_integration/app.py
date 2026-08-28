from prediction import predict_stress
from recommendation_engine import generate_recommendations


def main():

    print("=" * 50)
    print("       AI-BASED MENTAL STRESS PREDICTION")
    print("=" * 50)

    print("\nEnter Student Details\n")

    student = {}

    student["Age"] = input("Age (example: 18–20): ")
    student["Year of Study"] = input("Year of Study (example: 1st Year): ")
    student["How often do you feel stressed?"] = input(
        "How often do you feel stressed?: "
    )
    student["What is the main cause of your stress?"] = input(
        "Main cause of stress?: "
    )
    student["How many hours do you sleep on average?"] = input(
        "How many hours do you sleep on average?: "
    )
    student["How often do you exercise or do physical activity?"] = input(
        "How often do you exercise or do physical activity?: "
    )
    student["Have you experienced any of the following due to stress?"] = input(
        "Stress-related symptoms?: "
    )
    student["How do you usually cope with stress?"] = input(
        "How do you usually cope with stress?: "
    )

    # -------------------------------
    # PREDICTION
    # -------------------------------

    predicted_stress = predict_stress(student)

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)

    print("Predicted Stress Level:", predicted_stress)

    # -------------------------------
    # RECOMMENDATIONS
    # -------------------------------

    recommendations = generate_recommendations(student)

    print("\nRecommended Strategies:")
    
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")

    print("\n" + "=" * 50)
    print("Note: This prediction is based on survey data and")
    print("statistical modelling. It is not a medical diagnosis.")
    print("=" * 50)


if __name__ == "__main__":
    main()