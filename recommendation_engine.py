def generate_recommendations(student_data):
    recommendations = []

    # Sleep
    sleep = str(
        student_data.get("How many hours do you sleep on average?", "")
    ).lower()

    if any(value in sleep for value in ["less", "4", "5", "6"]):
        recommendations.append(
            "Try to maintain a consistent and sufficient sleep routine."
        )

    # Academic pressure / Exams
    stress_cause = str(
        student_data.get("What is the main cause of your stress?", "")
    ).lower()

    if "academic" in stress_cause or "exam" in stress_cause:
        recommendations.append(
            "Break academic work into smaller tasks and plan study time in advance."
        )

    # Exercise
    exercise = str(
        student_data.get(
            "How often do you exercise or do physical activity?", ""
        )
    ).lower()

    if any(value in exercise for value in ["rarely", "never", "sometimes"]):
        recommendations.append(
            "Include regular physical activity or short movement breaks in your routine."
        )

    # Stress symptoms
    symptoms = str(
        student_data.get(
            "Have you experienced any of the following due to stress?", ""
        )
    ).lower()

    if symptoms and symptoms not in ["none", "no", "nan"]:
        recommendations.append(
            "Take regular study breaks and use healthy coping strategies when you feel stressed."
        )

    # Coping method
    coping = str(
        student_data.get(
            "How do you usually cope with stress?", ""
        )
    ).lower()

    if coping in ["nothing", "none", ""]:
        recommendations.append(
            "Consider using healthy coping strategies such as talking with someone you trust, "
            "taking breaks, or doing a relaxing activity."
        )

    # Default recommendation
    if not recommendations:
        recommendations.append(
            "Continue maintaining healthy study, sleep, activity, and social routines."
        )

    return recommendations


if __name__ == "__main__":

    # Test student
    sample_student = {
        "What is the main cause of your stress?": "Academic pressure",
        "How many hours do you sleep on average?": "5-6 hours",
        "How often do you exercise or do physical activity?": "Rarely",
        "Have you experienced any of the following due to stress?": "Headache",
        "How do you usually cope with stress?": "Nothing"
    }

    recommendations = generate_recommendations(sample_student)

    print("\nRecommended Strategies:")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")