import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_job_role


EXPERIENCE_OPTIONS = {
    "No experience": "Без опыта",
    "1–3 years": "От 1 года до 3 лет",
    "3–6 years": "От 3 до 6 лет",
    "6+ years": "Опыт более 6 лет",
}

EDUCATION_OPTIONS = {
    "Not specified": "Not Specified",
    "Bachelor's (general)": "Высшее образование",
    "Bachelor's (technical)": "Высшее техническое образование",
    "Vocational": "Среднее специальное образование",
    "Student": "Студент",
    "Not relevant": "Не имеет значения",
}


st.set_page_config(
    page_title="Skill-Based Job Role Predictor",
    layout="centered",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        max-width: 720px;
    }

    .app-header {
        text-align: center;
        margin-bottom: 1.75rem;
    }
    .app-header h1 {
        font-weight: 700;
        font-size: 2.1rem;
        margin: 0 0 0.4rem 0;
        color: #111827;
        letter-spacing: -0.01em;
    }
    .app-header p {
        color: #6b7280;
        font-size: 1rem;
        margin: 0;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }

    .stButton > button, .stFormSubmitButton > button {
        background: #4f46e5;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: background 0.15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #4338ca;
        color: #ffffff;
    }

    .result-card {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 14px;
        padding: 1.75rem 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    .result-card .label {
        color: #4f46e5;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0;
        font-weight: 600;
    }
    .result-card .value {
        color: #1e1b4b;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.4rem 0 0 0;
        letter-spacing: -0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Skill-Based Job Role Predictor</h1>
        <p>Enter your skills and we will suggest the most fitting IT job role.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("predict_form", clear_on_submit=False):
    skills = st.text_area(
        "Your skills",
        placeholder="Python, SQL, pandas, statistics, Power BI",
        height=130,
    )
    col1, col2 = st.columns(2)
    with col1:
        experience_label = st.selectbox(
            "Experience level",
            list(EXPERIENCE_OPTIONS.keys()),
            index=1,
        )
    with col2:
        education_label = st.selectbox(
            "Education level",
            list(EDUCATION_OPTIONS.keys()),
            index=1,
        )
    submitted = st.form_submit_button("Predict job role", use_container_width=True)

if submitted:
    if not skills.strip():
        st.warning("Please enter at least one skill.")
    else:
        role = predict_job_role(
            skills=skills,
            experience_level=EXPERIENCE_OPTIONS[experience_label],
            education_level=EDUCATION_OPTIONS[education_label],
        )
        st.markdown(
            f"""
            <div class="result-card">
                <p class="label">Predicted role</p>
                <p class="value">{role}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
