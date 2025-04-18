import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import base64
import os

# Load model and features
model = joblib.load("asthma_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Set page configuration
st.set_page_config(page_title="Asthma Risk Predictor", layout="centered")

# Language toggle
lang = st.sidebar.selectbox("Language", ["English", "Hindi"])

texts = {
    "English": {
        "title": "🫁 Asthma Risk Prediction",
        "description": "Fill the following health parameters to check your risk of asthma.",
        "predict": "Predict",
        "input_preview": "Model Input Preview",
        "asthma": "⚠️ Asthma Detected. Please consult a doctor.",
        "no_asthma": "✅ No Asthma Detected. Keep up the healthy lifestyle!",
        "download_report": "📄 Download Report (PDF)"
    },
    "Hindi": {
        "title": "🫁 दमा जोखिम पूर्वानुमान",
        "description": "दमा के जोखिम की जांच के लिए निम्नलिखित स्वास्थ्य पैरामीटर भरें।",
        "predict": "पूर्वानुमान करें",
        "input_preview": "मॉडल इनपुट पूर्वावलोकन",
        "asthma": "⚠️ दमा का जोखिम पाया गया। कृपया डॉक्टर से संपर्क करें।",
        "no_asthma": "✅ कोई दमा नहीं पाया गया। स्वस्थ जीवनशैली बनाए रखें!",
        "download_report": "📄 रिपोर्ट डाउनलोड करें (PDF)"
    }
}

text = texts[lang]

st.title(text["title"])
st.markdown(text["description"])

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("""<style>body { background-color: #1e1e1e; color: white; }</style>""", unsafe_allow_html=True)

# Labels
pretty_labels = {
    "PollutionExposure": "Pollution Exposure (0–9)",
    "PollenExposure": "Pollen Exposure (0–9)",
    "SleepQuality": "Sleep Quality (0–9)",
    "PhysicalActivity": "Physical Activity (0–9)",
    "DustExposure": "Dust Exposure (0–9)",
    "DietQuality": "Diet Quality (0–9)"
}

# Collect input
user_input = {}
for feature in feature_names:
    if feature == "Age":
        user_input[feature] = st.slider("Age (years)", 5, 80, 25)
    elif feature in pretty_labels:
        user_input[feature] = st.slider(pretty_labels[feature], 0, 9, 5)
    else:
        user_input[feature] = st.selectbox(f"{feature}", [0, 1, 2, 3], format_func=lambda x: f"Level {x}")

# Prediction logic
if st.button(f"🔍 {text['predict']}"):
    input_df = pd.DataFrame([user_input])

    # ✅ Ensure correct feature order
    input_df = input_df[feature_names]

    st.write(f"✅ {text['input_preview']}:", input_df)

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Probability")
    st.write({
        "No Asthma": round(prediction_proba[0]*100, 2),
        "Asthma": round(prediction_proba[1]*100, 2)
    })

    # Lifestyle suggestions
    st.subheader("💡 Lifestyle Suggestions")
    if prediction == 1:
        st.error(text['asthma'])
        st.markdown("- 🧘 Improve air quality\n- 🛏️ Sleep better\n- 🥗 Eat healthy\n- 🏃 Be active")
    else:
        st.success(text['no_asthma'])
        st.markdown("- 👍 Keep it up!\n- 🌿 Avoid pollution\n- 💤 Maintain rest routines")

    # PDF Report
    pdf = FPDF()
    pdf.add_page()
    
    # Use Unicode-supporting font
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        st.warning("Please download 'DejaVuSans.ttf' and place it in the app directory for Unicode PDF support.")
    else:
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)

        pdf.cell(200, 10, txt=text['title'], ln=True, align='C')
        for key, val in user_input.items():
            pdf.cell(200, 10, txt=f"{key}: {val}", ln=True)

        pdf.cell(200, 10, txt=f"Prediction: {'Asthma' if prediction == 1 else 'No Asthma'}", ln=True)

        pdf_path = "asthma_report.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            b64_pdf = base64.b64encode(f.read()).decode('utf-8')
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="asthma_report.pdf">{text["download_report"]}</a>'
            st.markdown(href, unsafe_allow_html=True)
