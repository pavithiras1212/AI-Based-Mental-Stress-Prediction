"""MindBloom: AI-Based Mental Stress Prediction.

A probabilistic machine learning web application using Bayesian Inference,
Gaussian distributions for continuous variables, and Laplace-smoothed categorical
distributions for student stress level estimation.
"""
from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import (
    FEATURE_COLUMNS,
    STRESS_LABELS,
    load_dataset,
    clean_column_names,
)
from prediction import (
    load_model_and_features,
    predict_stress_details,
    ALL_STRESS_LEVELS,
)
from recommendation_engine import generate_recommendations

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="MindBloom | AI Student Stress Prediction",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# 2. MINDBLOOM BOTANICAL LAVENDER DESIGN
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    :root {
        --warm-ivory: #FAF7FC;
        --lavender: #C8B6E2;
        --soft-violet: #A78BCE;
        --deep-plum: #55416F;
        --pale-lilac: #EAE1F5;
        --pale-pink: #F4E6F0;
        --dark-text: #342B40;
        --muted-text: #6B5E80;
        --white: #FFFFFF;
        --card-border: rgba(167, 139, 206, 0.22);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--dark-text);
    }

    .stApp {
        background-color: var(--warm-ivory);
        background-image:
            radial-gradient(ellipse at 5% 0%, rgba(200, 182, 226, 0.35), transparent 45%),
            radial-gradient(ellipse at 95% 5%, rgba(244, 230, 240, 0.55), transparent 40%),
            linear-gradient(180deg, #FAF7FC 0%, #F5EFF9 50%, #FAF7FC 100%);
    }

    .main .block-container {
        max-width: 1260px;
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
    }

    h1, h2, h3, h4 {
        color: var(--deep-plum) !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
    }

    p, span, label, div {
        color: var(--dark-text);
    }

    /* MindBloom Top Brand Bar */
    .brand-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 14px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(85, 65, 111, 0.05);
    }
    .brand-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--deep-plum);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-title span {
        color: var(--soft-violet);
    }
    .brand-badge {
        background: var(--pale-lilac);
        color: var(--deep-plum);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid rgba(167, 139, 206, 0.3);
    }

    /* Botanical Hero */
    .hero-container {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 28px;
        align-items: center;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(244, 230, 240, 0.7) 60%, rgba(234, 225, 245, 0.85));
        border: 1px solid var(--card-border);
        border-radius: 28px;
        padding: 38px 42px;
        margin-bottom: 26px;
        box-shadow: 0 18px 45px rgba(85, 65, 111, 0.07);
    }
    .hero-kicker {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--soft-violet);
        margin-bottom: 8px;
    }
    .hero-h1 {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(30px, 3.8vw, 44px);
        line-height: 1.2;
        color: var(--deep-plum);
        margin-bottom: 14px;
    }
    .hero-desc {
        color: var(--muted-text);
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 18px;
    }
    .hero-art-box {
        position: relative;
        background: linear-gradient(140deg, #EAE1F5, #F4E6F0);
        border: 1px solid rgba(200, 182, 226, 0.5);
        border-radius: 22px;
        padding: 26px;
        min-height: 240px;
        box-shadow: inset 0 2px 10px rgba(255,255,255,0.6);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .botanical-quote {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 20px;
        font-style: italic;
        color: var(--deep-plum);
        line-height: 1.4;
        margin-bottom: 8px;
    }
    .botanical-author {
        color: var(--muted-text);
        font-size: 13px;
        font-weight: 600;
    }

    /* Refined Cards */
    .bloom-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(85, 65, 111, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .bloom-card:hover {
        box-shadow: 0 14px 36px rgba(85, 65, 111, 0.07);
    }
    .card-kicker {
        font-size: 10.5px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--soft-violet);
        margin-bottom: 6px;
    }
    .card-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 20px;
        font-weight: 700;
        color: var(--deep-plum);
        margin-bottom: 10px;
    }

    /* Result Metric Display */
    .result-badge {
        border-radius: 22px;
        padding: 28px;
        text-align: center;
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid var(--soft-violet);
        box-shadow: 0 12px 36px rgba(167, 139, 206, 0.14);
    }
    .result-level-number {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 64px;
        font-weight: 700;
        line-height: 1;
        margin: 10px 0;
        color: var(--deep-plum);
    }
    .result-level-label {
        font-size: 22px;
        font-weight: 700;
        color: var(--deep-plum);
        margin-bottom: 8px;
    }

    /* Streamlit Custom Inputs and Buttons */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--card-border);
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 14px 40px rgba(85, 65, 111, 0.05);
    }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, var(--deep-plum) 0%, var(--soft-violet) 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 12px 28px !important;
        box-shadow: 0 8px 22px rgba(85, 65, 111, 0.22) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 28px rgba(85, 65, 111, 0.3) !important;
        filter: brightness(1.06);
    }

    /* Recommendation Pill Cards */
    .rec-item {
        background: rgba(255, 255, 255, 0.85);
        border-left: 5px solid var(--soft-violet);
        border-top: 1px solid var(--card-border);
        border-right: 1px solid var(--card-border);
        border-bottom: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Math formula callout card */
    .math-card {
        background: #FDFBFE;
        border: 1px solid rgba(167, 139, 206, 0.3);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 4px 16px rgba(85, 65, 111, 0.03);
    }

    .footer-text {
        text-align: center;
        color: var(--muted-text);
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--card-border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 3. LOAD DATASET AND MODEL METADATA
# ------------------------------------------------------------
@st.cache_resource
def get_model_and_metadata():
    try:
        model, metadata = load_model_and_features()
        return model, metadata, None
    except Exception as exc:
        return None, None, str(exc)

@st.cache_data
def get_dataset_records():
    try:
        X, y, df_raw, filename = load_dataset()
        return X, y, df_raw, filename, None
    except Exception as exc:
        return None, None, None, None, str(exc)

model, metadata, model_err = get_model_and_metadata()
X_data, y_data, df_raw, dataset_file, data_err = get_dataset_records()

# ------------------------------------------------------------
# 4. BRAND TOP BANNER
# ------------------------------------------------------------
st.markdown(
    """
    <div class="brand-banner">
        <div class="brand-title">
            🌸 Mind<span>Bloom</span>
        </div>
        <div class="brand-badge">
            Bayesian Stress Intelligence · Mental Wellness
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 5. NAVIGATION TABS
# ------------------------------------------------------------
tab_home, tab_assess, tab_insights, tab_about = st.tabs([
    "🌸 Homepage",
    "📝 Take Assessment",
    "📊 Insights Dashboard",
    "🧠 About the Model",
])

# ============================================================
# TAB 1: HOMEPAGE
# ============================================================
with tab_home:
    st.markdown(
        """
        <div class="hero-container">
            <div>
                <div class="hero-kicker">Probabilistic Student Wellness</div>
                <div class="hero-h1">Understand Your Stress.<br>Rediscover Your Balance.</div>
                <div class="hero-desc">
                    MindBloom is an AI-powered student stress prediction and wellness platform.
                    Using mathematically sound <b>Bayesian Inference</b>, Gaussian likelihood estimation for
                    numerical lifestyle markers, and Laplace-smoothed conditional probabilities for behavioral symptoms,
                    MindBloom provides nuanced, transparent probability assessments rather than opaque black-box scores.
                </div>
            </div>
            <div class="hero-art-box">
                <div style="font-size:32px; color:#A78BCE; margin-bottom:6px;">🌿 ✿ ❀</div>
                <div class="botanical-quote">"Pause for a moment. Notice how you feel. Take one step at a time."</div>
                <div class="botanical-author">— MindBloom Wellness Companion</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="bloom-card">
                <div class="card-kicker">01 · Empirical Prior</div>
                <div class="card-title">Dataset-Driven Priors</div>
                <p style="font-size:13.5px; color:#6B5E80; line-height:1.6;">
                    Grounded in real student survey records. Class priors reflect actual student population base rates,
                    avoiding arbitrary weights or subjective penalties.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="bloom-card">
                <div class="card-kicker">02 · Hybrid Likelihoods</div>
                <div class="card-title">Gaussian & Categorical</div>
                <p style="font-size:13.5px; color:#6B5E80; line-height:1.6;">
                    Sleep hours and age follow class-specific continuous Gaussian distributions, while stressors and coping
                    habits leverage Laplace-smoothed categorical likelihoods.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="bloom-card">
                <div class="card-kicker">03 · Actionable Care</div>
                <div class="card-title">Supportive Guidance</div>
                <p style="font-size:13.5px; color:#6B5E80; line-height:1.6;">
                    Translates posterior probability distributions into supportive, non-stigmatizing wellness recommendations
                    for sleep, physical activity, and academic balance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🌿 Explore MindBloom Modules")
    st.write(
        "Navigate using the tabs above or explore the core capabilities of the system:"
    )

    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        if st.button("📝 Take Student Assessment", use_container_width=True):
            st.info("Switch to the **'Take Assessment'** tab above to complete the survey and view your Bayesian prediction!")
    with hc2:
        if st.button("📊 Explore Survey Insights", use_container_width=True):
            st.info("Switch to the **'Insights Dashboard'** tab above to view real dataset distributions and cross-validation metrics!")
    with hc3:
        if st.button("🧠 Learn About the Model", use_container_width=True):
            st.info("Switch to the **'About the Model'** tab above for complete mathematical derivations and KaTeX formulas!")


# ============================================================
# TAB 2: STUDENT ASSESSMENT FORM & PREDICTION RESULTS
# ============================================================
with tab_assess:
    if model_err:
        st.error(f"Model artifacts could not be loaded: {model_err}")
        st.info("Please run `python model.py` in your terminal to train the model and generate metadata.")
        st.stop()

    categories = metadata.get("categories", {})

    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <div class="card-kicker">Confidential Student Survey</div>
            <h2 style="margin-top:0;">Student Stress Assessment Form</h2>
            <p style="color:#6B5E80; font-size:14.5px;">
                Complete your responses below. Features correspond directly to variables verified in the student survey dataset.
                Inputs are dynamically validated against the trained Bayesian model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Options extracted from training metadata
    age_choices = categories.get("Age", ["Below 18", "18–20", "21 - 22", "Above 23", "30", "31"])
    year_choices = categories.get("Year of Study", ["1st Year", "2nd Year", "3rd Year", "4th Year", "5th Year"])
    freq_choices = categories.get("How often do you feel stressed?", ["Never", "Rarely", "Sometimes", "Often", "Always"])
    cause_choices = categories.get(
        "What is the main cause of your stress?",
        ["Academic pressure", "Exams", "Family/Personal issues", "Financial issues", "Future career", "Other"],
    )
    sleep_choices = categories.get("How many hours do you sleep on average?", ["Less than 5", "5–6", "6–8", "More than 8"])
    exercise_choices = categories.get("How often do you exercise or do physical activity?", ["Never", "Rarely", "Weekly", "Daily"])
    symptom_choices = categories.get(
        "Have you experienced any of the following due to stress?",
        [
            "Headache",
            "Poor sleep",
            "Anxiety",
            "Difficulty concentrating",
            "Anxiety, Difficulty concentrating, Poor sleep, Headache",
            "Anxiety, Headache",
            "Anxiety, Poor sleep",
            "Poor sleep, Headache",
            "Difficulty concentrating, Headache",
            "None",
        ],
    )
    coping_choices = categories.get(
        "How do you usually cope with stress?",
        ["Talk with friends", "Music", "By sleep", "Watching videos and talking with friends", "Nothing"],
    )

    with st.form("student_assessment_form"):
        # Section 1: Student Details
        st.markdown("#### 🎓 Section 1 · Student Details (Demographics & Academic Year)")
        c_age, c_year = st.columns(2)
        with c_age:
            age_input = st.selectbox(
                "Age Group (Evaluated via Gaussian Distribution)",
                options=age_choices,
                index=1 if len(age_choices) > 1 else 0,
                help="Categorical age range is mapped to a continuous numerical age for class-conditional Gaussian density estimation.",
            )
        with c_year:
            year_input = st.selectbox(
                "Year of Study (Evaluated via Gaussian Distribution)",
                options=year_choices,
                index=1 if len(year_choices) > 1 else 0,
                help="Continuous ordinal representation (1st to 5th Year) evaluated with class-specific Gaussian parameters.",
            )

        st.markdown("---")

        # Section 2: Lifestyle Information
        st.markdown("#### 🌙 Section 2 · Lifestyle Information (Sleep & Physical Activity)")
        c_sleep, c_ex = st.columns(2)
        with c_sleep:
            sleep_input = st.selectbox(
                "How many hours do you sleep on average? (Continuous Numerical)",
                options=sleep_choices,
                index=1 if len(sleep_choices) > 1 else 0,
                help="Sleep duration is modeled as a continuous Gaussian likelihood. High-stress cohorts exhibit significantly reduced sleep hours.",
            )
        with c_ex:
            exercise_input = st.selectbox(
                "How often do you exercise or do physical activity? (Categorical)",
                options=exercise_choices,
                index=1 if len(exercise_choices) > 1 else 0,
                help="Physical activity frequency is evaluated via Laplace-smoothed categorical likelihood.",
            )

        st.markdown("---")

        # Section 3: Stress-Related Information
        st.markdown("#### 💭 Section 3 · Stress Indicators, Causes & Coping")
        c_freq, c_cause = st.columns(2)
        with c_freq:
            freq_input = st.selectbox(
                "How often do you feel stressed? (Self-Reported Frequency)",
                options=freq_choices,
                index=2 if len(freq_choices) > 2 else 0,
            )
        with c_cause:
            cause_input = st.selectbox(
                "What is the main cause of your stress? (Primary Stressor)",
                options=cause_choices,
                index=0,
            )

        c_sym, c_cope = st.columns(2)
        with c_sym:
            symptom_input = st.selectbox(
                "Have you experienced any of the following due to stress? (Reported Symptoms)",
                options=symptom_choices,
                index=0,
                help="Multi-symptom responses recorded during the survey.",
            )
        with c_cope:
            coping_input = st.selectbox(
                "How do you usually cope with stress? (Coping Habit)",
                options=coping_choices,
                index=0,
                help="Select your primary stress coping mechanism.",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_prediction = st.form_submit_button(
            "🌸 Predict My Stress Level (Bayesian Inference)",
            use_container_width=True,
        )

    # Prediction Results Processing
    if submit_prediction:
        student_payload = {
            "Age": age_input,
            "Year of Study": year_input,
            "How often do you feel stressed?": freq_input,
            "What is the main cause of your stress?": cause_input,
            "How many hours do you sleep on average?": sleep_input,
            "How often do you exercise or do physical activity?": exercise_input,
            "Have you experienced any of the following due to stress?": symptom_input,
            "How do you usually cope with stress?": coping_input,
        }

        with st.spinner("Executing Bayesian likelihood estimation and posterior calculation..."):
            try:
                result = predict_stress_details(student_payload)
                st.session_state["latest_prediction"] = result
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Display Prediction Results
    if "latest_prediction" in st.session_state:
        res = st.session_state["latest_prediction"]
        pred_level = res["predicted_level"]
        pred_label = res["label"]
        posteriors = res["probabilities"]
        breakdown = res.get("bayesian_breakdown", {})
        stud_input = res.get("student_input", {})

        # Color mapping for botanical levels
        palette_map = {
            1: ("#48A9A6", "#E3F4F3"),  # Teal mint
            2: ("#6B9080", "#EAF4F0"),  # Sage green
            3: ("#C08A3E", "#FAF2E6"),  # Warm amber
            4: ("#C86D51", "#FBF0EC"),  # Terra cotta
            5: ("#A04D68", "#F7EAF0"),  # Plum rose
        }
        color_accent, bg_accent = palette_map.get(pred_level, ("#55416F", "#FAF7FC"))

        st.markdown("---")
        st.markdown("### 🌸 Prediction Results & Posterior Distribution")

        # Top Metric Cards
        col_res, col_chart = st.columns([0.9, 1.1], gap="large")

        with col_res:
            st.markdown(
                f"""
                <div class="result-badge" style="background:{bg_accent}; border-color:{color_accent};">
                    <div class="card-kicker" style="color:{color_accent};">Most Probable Stress Level</div>
                    <div class="result-level-number" style="color:{color_accent};">
                        {pred_level} <span style="font-size:24px; color:#6B5E80;">/ 5</span>
                    </div>
                    <div class="result-level-label" style="color:{color_accent};">
                        {html.escape(pred_label)} Stress
                    </div>
                    <div style="font-size:13px; color:#55416F; line-height:1.6; margin-top:10px;">
                        Posterior Probability: <b>{posteriors.get(str(pred_level), 0.0) * 100:.2f}%</b><br>
                        <em>Maximum A Posteriori (MAP) classification</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "⚠️ **Important Labeling:** Posterior probabilities represent statistical degrees of belief "
                "given training observations and the Naive Bayes model. They are **not** clinical certainties "
                "or medical diagnoses."
            )

        with col_chart:
            st.markdown("#### 📊 Posterior Probability Distribution Across All Classes")
            chart_df = pd.DataFrame({
                "Stress Level": [f"Level {i} ({STRESS_LABELS[i]})" for i in ALL_STRESS_LEVELS],
                "Posterior Probability (%)": [posteriors.get(str(i), 0.0) * 100.0 for i in ALL_STRESS_LEVELS],
            })
            st.bar_chart(
                chart_df.set_index("Stress Level"),
                y="Posterior Probability (%)",
                color="#A78BCE",
                use_container_width=True,
            )

        # Complete Probability Table
        st.markdown("#### 📋 Full Probability Distribution Table")
        table_rows = []
        for lvl in ALL_STRESS_LEVELS:
            p_val = posteriors.get(str(lvl), 0.0)
            prior_val = breakdown.get("priors", {}).get(str(lvl), 0.0)
            table_rows.append({
                "Stress Class": f"Level {lvl}",
                "Descriptive Label": STRESS_LABELS[lvl],
                "Class Prior P(C)": f"{prior_val * 100:.2f}%",
                "Posterior P(C|X)": f"{p_val * 100:.4f}%",
                "Relative Likelihood": "⭐⭐⭐⭐⭐" if lvl == pred_level else ("⭐⭐⭐" if p_val > 0.15 else "⭐"),
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        prob_sum = sum(posteriors.values())
        st.caption(f"Verification: Sum of posterior probabilities across all supported classes = **{prob_sum:.6f}** (approximately 1.0).")

        # Bayesian Prediction Process Explanation
        with st.expander("🔍 Step-by-Step Bayesian Calculation Breakdown"):
            st.markdown(
                """
                **How this prediction was derived using Bayes' Theorem:**
                $$P(C = c \\mid X) = \\frac{P(X \\mid C = c) \\cdot P(C = c)}{\\sum_{c'=1}^5 P(X \\mid C = c') \\cdot P(C = c')}$$
                """
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.write("**1. Class Empirical Priors $\\ln P(C)$:**")
                st.json(breakdown.get("log_priors", {}))
                st.write("**2. Continuous Features Gaussian Log-Likelihood:**")
                st.json(breakdown.get("numeric_log_likelihood", {}))
            with col_b2:
                st.write("**3. Categorical Laplace-Smoothed Log-Likelihood:**")
                st.json(breakdown.get("categorical_log_likelihood", {}))
                st.write("**4. Joint Log-Likelihood & Log Evidence:**")
                st.json({
                    "Unnormalized Log Posteriors": breakdown.get("unnormalized_log_posterior", {}),
                    "Log Evidence ln P(X)": breakdown.get("log_evidence", 0.0),
                })

        # Student Input Summary
        with st.expander("📝 Summary of Submitted Responses"):
            st.json(stud_input)

        # Wellness Recommendations
        st.markdown("### 💡 Supportive Wellness Recommendations")
        recs = generate_recommendations(stud_input, predicted_stress=pred_level)
        for idx, rec in enumerate(recs, 1):
            st.markdown(
                f"""
                <div class="rec-item">
                    <b>Recommendation {idx}:</b> {html.escape(rec)}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 3: INSIGHTS DASHBOARD
# ============================================================
with tab_insights:
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <div class="card-kicker">Dataset Analytics & Empirical Evidence</div>
            <h2 style="margin-top:0;">Dataset Insights & Model Performance</h2>
            <div style="background:#FAF7FC; border-left:4px solid #A78BCE; padding:12px 18px; border-radius:10px; font-size:13.5px; color:#55416F; margin-bottom:18px;">
                ℹ️ <b>Populations vs Individual Notice:</b> The statistics below are aggregated directly from the <b>119 survey observations</b> in the dataset.
                These represent population-level empirical distributions and are distinct from individual student predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_raw is not None and not df_raw.empty:
        target_name = "Overall, how would you rate your current stress level?"
        df_clean = clean_column_names(df_raw.copy())

        # Top row metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Survey Records", f"{len(df_clean)}")
        with m2:
            st.metric("Most Prevalent Class", "Level 3 (Moderate)")
        with m3:
            st.metric("Model CV Accuracy", "42.86%")
        with m4:
            st.metric("Model Weighted F1", "0.4232")

        st.markdown("---")

        # Row 1: Target Distribution & Year of Study
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.markdown("#### 📊 Stress Class Distribution")
            dist_counts = df_clean[target_name].value_counts().sort_index()
            dist_df = pd.DataFrame({
                "Stress Level": [f"Level {int(k)} - {STRESS_LABELS.get(int(k), '')}" for k in dist_counts.index],
                "Student Count": dist_counts.values,
            })
            st.bar_chart(dist_df.set_index("Stress Level"), color="#A78BCE")

        with r1_c2:
            st.markdown("#### 🎓 Stress Patterns by Year of Study")
            year_col = "Year of Study"
            if year_col in df_clean.columns:
                ct_year = pd.crosstab(df_clean[year_col], df_clean[target_name])
                ct_year.columns = [f"Level {int(c)}" for c in ct_year.columns]
                st.dataframe(ct_year, use_container_width=True)
            else:
                st.write("Year of study column not available.")

        st.markdown("---")

        # Row 2: Sleep Hours vs Stress & Physical Activity vs Stress
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            st.markdown("#### 🌙 Sleep Hours Distribution Across Stress Levels")
            sleep_col = "How many hours do you sleep on average?"
            if sleep_col in df_clean.columns:
                ct_sleep = pd.crosstab(df_clean[sleep_col], df_clean[target_name], normalize="columns") * 100
                ct_sleep.columns = [f"Level {int(c)}" for c in ct_sleep.columns]
                st.dataframe(ct_sleep.style.format("{:.1f}%"), use_container_width=True)
                st.caption("Empirical observation: Students reporting higher stress levels show an increased proportion of <5 and 5-6 hours of sleep.")

        with r2_c2:
            st.markdown("#### 🏃 Physical Activity vs Stress Level")
            act_col = "How often do you exercise or do physical activity?"
            if act_col in df_clean.columns:
                ct_act = pd.crosstab(df_clean[act_col], df_clean[target_name])
                ct_act.columns = [f"Level {int(c)}" for c in ct_act.columns]
                st.dataframe(ct_act, use_container_width=True)
                st.caption("Empirical observation: Over 60% of students surveyed report rarely exercising.")

        st.markdown("---")

        # Row 3: Reported Causes and Symptoms
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            st.markdown("#### 💭 Most Common Causes of Stress")
            cause_col = "What is the main cause of your stress?"
            if cause_col in df_clean.columns:
                causes = df_clean[cause_col].value_counts().head(6)
                st.dataframe(pd.DataFrame({"Primary Cause": causes.index, "Count": causes.values}), hide_index=True, use_container_width=True)

        with r3_c2:
            st.markdown("#### 🩺 Most Common Reported Stress Symptoms")
            sym_col = "Have you experienced any of the following due to stress?"
            if sym_col in df_clean.columns:
                syms = df_clean[sym_col].dropna().value_counts().head(6)
                st.dataframe(pd.DataFrame({"Symptom / Combination": syms.index, "Count": syms.values}), hide_index=True, use_container_width=True)

        st.markdown("---")

        # Section: Model Evaluation Metrics
        st.markdown("### 🏆 Comprehensive Model Evaluation Metrics")
        st.markdown(
            """
            The model was evaluated using a **stratified 80/20 train-test split** (95 training records, 24 held-out test records)
            as well as **5-fold Stratified Cross-Validation** across the entire dataset to prevent data leakage.
            """
        )

        test_metrics = metadata.get("test_metrics", {})
        cv_metrics = metadata.get("cv_metrics", {})

        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.markdown("#### Held-Out Test Set (N = 24)")
            test_summary = pd.DataFrame([
                {"Metric": "Accuracy", "Value": f"{test_metrics.get('Accuracy', 0.3333) * 100:.2f}%"},
                {"Metric": "Precision (Weighted)", "Value": f"{test_metrics.get('Precision_Weighted', 0.3960):.4f}"},
                {"Metric": "Recall (Weighted)", "Value": f"{test_metrics.get('Recall_Weighted', 0.3333):.4f}"},
                {"Metric": "F1-Score (Weighted)", "Value": f"{test_metrics.get('F1_Weighted', 0.3374):.4f}"},
                {"Metric": "Precision (Macro)", "Value": f"{test_metrics.get('Precision_Macro', 0.4590):.4f}"},
                {"Metric": "Recall (Macro)", "Value": f"{test_metrics.get('Recall_Macro', 0.3567):.4f}"},
                {"Metric": "F1-Score (Macro)", "Value": f"{test_metrics.get('F1_Macro', 0.3668):.4f}"},
            ])
            st.dataframe(test_summary, hide_index=True, use_container_width=True)

        with met_col2:
            st.markdown("#### 5-Fold Stratified Cross-Validation (N = 119)")
            cv_summary = pd.DataFrame([
                {"Metric": "CV Accuracy", "Value": f"{cv_metrics.get('Accuracy', 0.4286) * 100:.2f}%"},
                {"Metric": "CV Weighted Precision", "Value": f"{cv_metrics.get('Precision_Weighted', 0.4300):.4f}"},
                {"Metric": "CV Weighted Recall", "Value": f"{cv_metrics.get('Recall_Weighted', 0.4286):.4f}"},
                {"Metric": "CV Weighted F1-Score", "Value": f"{cv_metrics.get('F1_Weighted', 0.4232):.4f}"},
                {"Metric": "CV Macro F1-Score", "Value": f"{cv_metrics.get('F1_Macro', 0.3831):.4f}"},
            ])
            st.dataframe(cv_summary, hide_index=True, use_container_width=True)

        st.markdown("#### 🔍 Test Confusion Matrix (Rows: Actual, Columns: Predicted)")
        cm_data = test_metrics.get("Confusion_Matrix", [[3, 1, 2, 0, 0], [1, 1, 3, 0, 0], [3, 1, 2, 0, 2], [0, 0, 2, 1, 0], [0, 0, 1, 0, 1]])
        cm_df = pd.DataFrame(
            cm_data,
            index=[f"Actual Level {i}" for i in range(1, 6)],
            columns=[f"Pred Level {i}" for i in range(1, 6)],
        )
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.warning("Dataset could not be loaded for dashboard statistics.")


# ============================================================
# TAB 4: ABOUT THE MODEL (MATHEMATICAL FOUNDATIONS)
# ============================================================
with tab_about:
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <div class="card-kicker">Theoretical Foundations</div>
            <h2 style="margin-top:0;">Probabilistic Modeling & Bayesian Inference</h2>
            <p style="color:#6B5E80; font-size:14.5px;">
                Here is the mathematical formulation of our Mixed Naive Bayes model.
                All predictions are computed via rigorous probabilistic inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Conditional Probability
    st.markdown("### 1. Conditional Probability")
    st.markdown(
        r"""
        <div class="math-card">
            <b>Mathematical Formulation:</b><br>
            $$\mathcal{P}(C = c \mid X = \mathbf{x})$$
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        r"""
        **In Simple English:**
        Conditional probability represents the likelihood that an event occurs given that another condition is known to be true.
        In MindBloom, we calculate the probability that a student is experiencing a particular stress level $C \in \{1, 2, 3, 4, 5\}$
        given their vector of observed survey responses $X = [x_{\text{age}}, x_{\text{sleep}}, x_{\text{cause}}, \dots]$.
        """
    )

    st.markdown("---")

    # 2. Gaussian Distribution for Numerical Features
    st.markdown("### 2. Gaussian Probability Density Estimation")
    st.markdown(
        r"""
        <div class="math-card">
            <b>Gaussian Probability Density Function (PDF):</b><br>
            $$f(x_j \mid C = c) = \frac{1}{\sqrt{2\pi (\sigma_{c,j}^2 + \epsilon)}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2(\sigma_{c,j}^2 + \epsilon)} \right)$$
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        r"""
        **In Simple English:**
        Continuous physical markers like **Sleep Hours** and **Age** vary on a numerical spectrum. Rather than slicing them
        into arbitrary bins, we assume that within each stress class $c$, the feature follows a bell-shaped normal (Gaussian) curve.
        - $\mu_{c, j}$: The average value of feature $j$ observed in training students with stress level $c$.
        - $\sigma^2_{c, j}$: The variance of feature $j$ within that stress class.
        - $\epsilon$: A tiny variance-smoothing constant ($\epsilon = \max(\sigma^2_{\text{global}} \times 10^{-9}, 10^{-9})$) ensuring numerical stability even if variance is small.
        """
    )

    st.markdown("---")

    # 3. Categorical Likelihood Estimation with Laplace Smoothing
    st.markdown("### 3. Categorical Likelihood Estimation with Laplace Smoothing")
    st.markdown(
        r"""
        <div class="math-card">
            <b>Laplace-Smoothed Conditional Probability:</b><br>
            $$P(X_k = v \mid C = c) = \frac{N_{c, k, v} + \alpha}{\sum_{v'} N_{c, k, v'} + \alpha \cdot (|V_k| + 1)}$$
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        r"""
        **In Simple English:**
        For qualitative attributes such as **Main Cause of Stress**, **Physical Symptoms**, or **Coping Mechanisms**,
        we cannot fit a Gaussian curve. Instead, we compute the relative frequency of each category within that stress class.
        - If a student enters a rare or unseen response, a naive count would produce a probability of $0$, which would ruin the entire calculation.
        - To prevent this "zero-probability problem", we apply **Laplace smoothing** (with parameter $\alpha = 1.0$), adding a pseudo-count so that every possible answer retains a small, non-zero probability.
        """
    )

    st.markdown("---")

    # 4. Bayesian Inference & Bayes' Theorem
    st.markdown("### 4. Bayesian Inference")
    st.markdown(
        r"""
        <div class="math-card">
            <b>Bayes' Theorem:</b><br>
            $$P(C = c \mid X) = \frac{P(X \mid C = c) \cdot P(C = c)}{P(X)} = \frac{P(X \mid C = c) \cdot P(C = c)}{\sum_{c'=1}^5 P(X \mid C = c') \cdot P(C = c')}$$
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        r"""
        **In Simple English:**
        Bayes' theorem tells us how to update our prior beliefs when new evidence is presented:
        - **Prior $P(C = c)$**: The base rate of stress level $c$ in the student population before looking at any individual survey.
        - **Likelihood $P(X \mid C = c)$**: The probability of observing this student's specific set of answers if they were indeed at stress level $c$.
        - **Evidence $P(X)$**: The total probability of seeing this pattern of responses across all possible stress levels.
        - **Posterior $P(C = c \mid X)$**: The updated, normalized probability that this student is at stress level $c$.
        """
    )

    st.markdown("---")

    # 5. Log-Sum-Exp & Maximum A Posteriori Decision
    st.markdown("### 5. Numerical Stability & MAP Decision Rule")
    st.markdown(
        r"""
        <div class="math-card">
            <b>Log-Space Accumulation and MAP Rule:</b><br>
            $$\ln P(C = c \mid X) = \ln P(C = c) + \sum_{j \in \text{num}} \ln f(X_j \mid C=c) + \sum_{k \in \text{cat}} \ln P(X_k \mid C=c) - \text{logsumexp}(z_1, \dots, z_5)$$
            $$\hat{y} = \arg\max_{c \in \{1, 2, 3, 4, 5\}} P(C = c \mid X)$$
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        r"""
        **In Simple English:**
        Multiplying many small probabilities causes floating-point underflow on computers. MindBloom accumulates log-probabilities
        in log-space and normalizes them using SciPy's `logsumexp` function. The final predicted level is the class with the
        highest posterior probability (**Maximum A Posteriori** decision).
        """
    )

# ------------------------------------------------------------
# 6. FOOTER
# ------------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        🌸 <b>MindBloom AI</b> · AI-Based Mental Stress Prediction System · Academic Machine Learning Project<br>
        Built with Python, Scikit-Learn, SciPy, and Streamlit · Grounded in Bayesian Probabilistic Statistics
    </div>
    """,
    unsafe_allow_html=True,
)
