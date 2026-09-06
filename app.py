import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Compatibility fix for models serialized with scikit-learn 1.6.x
import sklearn.compose._column_transformer as column_transformer
if not hasattr(column_transformer, "_RemainderColsList"):
    column_transformer._RemainderColsList = type(
        "_RemainderColsList", (list,), {}
    )

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_model.pkl"

st.set_page_config(
    page_title="Mental Health Score Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Put it in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)

TOP_COUNTRIES = [
    "Other", "India", "USA", "Canada", "Australia", "UK", "Germany",
    "Mexico", "Turkey", "France"
]

PLATFORMS = [
    "Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter", "Youtube",
    "Tiktok", "LINE", "KakaoTalk", "VKontakte", "Whatsapp", "WeChat"
]

PURPOSES = ["Networking", "Education", "Entertainment", "News"]
STRESS_LEVELS = ["Low", "Medium", "High", "Very High"]

try:
    model = load_model()
except Exception as exc:
    st.error("Unable to load the trained model.")
    st.code(str(exc))
    st.info("Keep app.py and Mental_Health_model.pkl in the same folder and use the provided compatibility code.")
    st.stop()

# ---------- Header ----------
st.markdown("# 🧠 Mental Health Score Predictor")
st.markdown(
    "Enter the student's lifestyle, academic, and social-media information to generate a predicted mental-health score."
)
st.caption("Machine-learning demo • Random Forest regression model")

st.divider()

with st.form("prediction_form"):
    st.subheader("👤 Personal & Academic Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=0, max_value=100, value=21, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with c2:
        country = st.selectbox("Country", TOP_COUNTRIES, index=1)
        academic_level = st.selectbox(
            "Academic Level",
            ["Undergraduate", "Graduate", "High School"]
        )

    with c3:
        platform = st.selectbox("Most Used Platform", PLATFORMS)
        purpose = st.selectbox("Purpose of Use", PURPOSES)

    st.subheader("📱 Usage & Lifestyle")
    c4, c5, c6 = st.columns(3)

    with c4:
        usage_hours = st.number_input(
            "Average Daily Usage (hours)",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5,
        )
        daily_unlocks = st.number_input(
            "Daily Unlocks",
            min_value=0,
            value=50,
            step=1,
        )

    with c5:
        study_hours = st.number_input(
            "Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5
        )
        physical_activity = st.number_input(
            "Physical Activity (hours)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )

    with c6:
        sleep_hours = st.number_input(
            "Sleep Hours per Night",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
        )
        stress_level = st.selectbox("Stress Level", STRESS_LEVELS, index=1)

    st.divider()
    submitted = st.form_submit_button("🔮 Predict Mental Health Score", use_container_width=True)

if submitted:
    grouped_country = country if country in TOP_COUNTRIES else "Other"

    # These names match the features stored in the trained pipeline.
    input_row = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Age": age,
        "Avg_Daily_Usage_Hours": usage_hours,
        "Daily_Unlocks": daily_unlocks,
        "Sleep_Hours_Per_Night": sleep_hours,
        "Physical_Activity_Hours": physical_activity,
        "Stress_Level": stress_level,
        "Gender": gender,
        "Academic_Level": academic_level,
        "Most_Used_Platform": platform,
        "Purpose_Of_Use": purpose,
        "Grouped_country": grouped_country,
    }])

    try:
        prediction = float(model.predict(input_row)[0])
        prediction = round(prediction, 2)

        st.success("Prediction generated successfully!")
        st.markdown("### Predicted Mental Health Score")
        st.metric("Score", f"{prediction:.2f}")

        st.progress(min(max(prediction / 100, 0.0), 1.0))
        st.caption(
            "The progress bar is only a visual representation. Do not interpret the score as a medical diagnosis."
        )

        with st.expander("View model input"):
            st.dataframe(input_row, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error("Prediction failed. Check that the model and its preprocessing environment are compatible.")
        st.exception(exc)

st.divider()
st.caption(
    "Educational project only. This prediction should not be used as a clinical diagnosis or as a substitute for professional mental-health care."
)
