"""Recommendation engine for MindBloom AI Stress Prediction.

Generates compassionate, actionable wellness recommendations tailored to
the student's predicted stress level and specific survey responses.
"""


def generate_recommendations(student_data, predicted_stress=None):
    """Generate recommendations based on predicted stress level and student survey responses.

    Parameters
    ----------
    student_data : dict
        Student survey responses.
    predicted_stress : int or float, optional
        Predicted stress level (1-5).

    Returns
    -------
    list of str
        Tailored wellness recommendations.
    """
    try:
        level = int(round(float(predicted_stress))) if predicted_stress is not None else None
    except (TypeError, ValueError):
        level = None

    recommendations = []

    # 1. Level-Specific Foundational Strategies
    if level == 1:
        recommendations.extend([
            "Your stress level is currently very low. Continue maintaining your balanced daily routine.",
            "Protect your restorative sleep and prioritize regular leisure and creative hobbies.",
            "Continue utilizing positive coping habits and sharing good moments with friends or family.",
        ])
    elif level == 2:
        recommendations.extend([
            "Your stress level is low. Keep practicing proactive time management to stay ahead of deadlines.",
            "Maintain a steady sleep schedule and incorporate regular physical activity into your week.",
            "Take short relaxation breaks between intense study sessions to sustain your focus.",
        ])
    elif level == 3:
        recommendations.extend([
            "Your stress level is moderate. Break large assignments into smaller, time-boxed milestones.",
            "Integrate restorative relaxation techniques such as deep diaphragmatic breathing (box breathing) or mindfulness.",
            "Review your weekly schedule to ensure you are leaving room for adequate sleep, nutrition, and exercise.",
        ])
    elif level == 4:
        recommendations.extend([
            "Your stress level is elevated. Consider deprioritizing non-essential commitments to reduce academic overload.",
            "Reach out to someone you trust—a friend, family member, mentor, or campus advisor—to share what you are experiencing.",
            "Focus on foundational recovery: set a firm bedtime, minimize late-night screen exposure, and take restorative walks.",
        ])
    elif level == 5:
        recommendations.extend([
            "Your stress level is very high. Please consider connecting with a university wellness counselor or student support service.",
            "Speak openly with academic advisors or faculty about workload adjustments or extension options.",
            "Prioritize basic self-care: consistent meals, hydration, and short moments of calming rest throughout the day.",
        ])
    else:
        recommendations.extend([
            "Maintain a balanced daily routine with adequate sleep, regular physical activity, and social connection.",
            "Break demanding academic tasks into smaller, manageable steps with planned breaks.",
        ])

    # 2. Targeted Factor-Specific Strategies
    normalized = {
        str(k).strip().casefold(): str(v).strip().casefold()
        for k, v in student_data.items()
    }

    sleep_text = normalized.get("how many hours do you sleep on average?", "")
    exercise_text = normalized.get("how often do you exercise or do physical activity?", "")
    cause_text = normalized.get("what is the main cause of your stress?", "")
    freq_text = normalized.get("how often do you feel stressed?", "")
    symptom_text = normalized.get("have you experienced any of the following due to stress?", "")
    coping_text = normalized.get("how do you usually cope with stress?", "")

    # Sleep intervention
    if any(term in sleep_text for term in ["less than 5", "<5", "4", "5–6", "5-6"]):
        recommendations.append(
            "Sleep Optimization: You reported sleeping fewer than 6 hours. Try keeping a consistent sleep schedule and stopping screen use 30 minutes before bed."
        )

    # Physical activity intervention
    if any(term in exercise_text for term in ["never", "rarely"]):
        recommendations.append(
            "Gentle Movement: Incorporating even 15-20 minutes of daily walking or light stretching significantly lowers cortisol levels and boosts mental clarity."
        )

    # Academic pressure intervention
    if any(term in cause_text for term in ["academic", "exam"]):
        recommendations.append(
            "Study Strategy: Use active study techniques like the Pomodoro technique (25 min study, 5 min break) to manage academic pressure without burnout."
        )

    # High stress frequency
    if freq_text in ["often", "always"]:
        recommendations.append(
            "Proactive Support: Because you experience frequent stress, setting up regular check-ins with peer support groups or mentors can make a meaningful difference."
        )

    # Physical symptoms reported
    if any(sym in symptom_text for sym in ["headache", "poor sleep", "anxiety", "difficulty concentrating"]):
        recommendations.append(
            "Physical Reset: When noticing headaches or anxiety, practice progressive muscle relaxation or step outside for fresh air to relieve tension."
        )

    # Ineffective or missing coping strategies
    if any(cop in coping_text for cop in ["nothing", "none", "", "nan"]):
        recommendations.append(
            "Healthy Coping: Build an active stress buffer—listening to music, journaling your thoughts, or going for a bike ride helps decompress."
        )

    # Deduplicate while preserving order
    unique_recs = []
    seen = set()
    for rec in recommendations:
        key = rec[:35].lower()
        if key not in seen:
            unique_recs.append(rec)
            seen.add(key)

    return unique_recs[:5]


if __name__ == "__main__":
    sample = {
        "Age": "18–20",
        "Year of Study": "2nd Year",
        "How often do you feel stressed?": "Often",
        "What is the main cause of your stress?": "Academic pressure",
        "How many hours do you sleep on average?": "Less than 5",
        "How often do you exercise or do physical activity?": "Never",
        "Have you experienced any of the following due to stress?": "Headache, Anxiety",
        "How do you usually cope with stress?": "Nothing",
    }
    recs = generate_recommendations(sample, predicted_stress=4)
    print("Sample Recommendations (Level 4):")
    for i, r in enumerate(recs, 1):
        print(f" {i}. {r}")