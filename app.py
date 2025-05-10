import streamlit as st
import google.generativeai as genai
from groq import Groq
from PIL import Image
import PyPDF2
import tempfile
import os
from google.api_core import exceptions
import time
import graphviz  # 🔄 NEW IMPORT
import base64
import subprocess  # 🔄 NEW IMPORT

# Hardcoded API keys
GEMINI_API_KEY = "AIzaSyCrjgJviN3ve3MnY8cjd6h2GXGS4Yp-Sp4"
GROQ_API_KEY = "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"

def configure_clients():
    genai.configure(api_key=GEMINI_API_KEY)
    global groq_client
    groq_client = Groq(api_key=GROQ_API_KEY)

configure_clients()

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# 🔄 Update background styling
def add_bg_from_local():
    css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffe6e6, #ff1493); /* Light pink to dark pink */
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background
add_bg_from_local()

def analyze_medical_report(content, content_type):
    prompt = "Analyze this medical report concisely. Provide key findings, diagnoses, and recommendations:"
    for attempt in range(MAX_RETRIES):
        try:
            if content_type == "image":
                response = gemini_model.generate_content([prompt, content])
            else:  # text
                response = gemini_model.generate_content(f"{prompt}\n\n{content}")
            return response.text
        except exceptions.GoogleAPIError as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze. Error: {str(e)}")
                return fallback_analysis(content, content_type)

def fallback_analysis(content, content_type):
    st.warning("Using fallback analysis due to API issues.")
    if content_type == "image":
        return "Unable to analyze the image. Please try again later or consult a medical professional."
    else:
        word_count = len(content.split())
        return f"""
        Fallback Analysis:
        1. Document Type: Text-based medical report
        2. Word Count: Approximately {word_count} words
        3. Recommendation: Review manually or consult a doctor.
        """

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def chat_with_report(question, analysis_text):
    messages = [
        {"role": "system", "content": "You are a knowledgeable medical assistant."},
        {"role": "system", "content": f"Report Analysis:\n{analysis_text}"},
        {"role": "user", "content": question}
    ]
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return resp.choices[0].message.content

def find_doctors(specialty, location):
    prompt = (
        f"List the top 5 {specialty} specialists in {location}. "
        "For each, provide name, clinic/hospital, address, and contact number."
    )
    messages = [{"role": "user", "content": prompt}]
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return resp.choices[0].message.content

def generate_flowchart_from_analysis(analysis_text):
    prompt = (
        "Based on this medical analysis, generate a step-by-step flowchart using simple terminology. "
        "Each step should represent a stage in diagnosis, findings, or recommendation. "
        "Use the format: Start -> Step1 -> Step2 -> ... -> End.\n\n"
        f"Analysis:\n{analysis_text}"
    )
    messages = [{"role": "user", "content": prompt}]
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return resp.choices[0].message.content

def render_flowchart(flow_text):
    steps = [s.strip() for s in flow_text.split("->")]
    dot = graphviz.Digraph()
    for i, step in enumerate(steps):
        dot.node(f"{i}", step)
        if i > 0:
            dot.edge(f"{i-1}", f"{i}")
    return dot

def main():
    st.markdown(
        """
        <h1 style='color: #ff1493;'>🩺 AI-driven Medical Report Analyser & Specialist Finder</h1>
        """,
        unsafe_allow_html=True
    )
    st.write("Upload a medical report (image or PDF), get AI analysis, ask questions, and find relevant specialists.")

    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'specialty' not in st.session_state:
        st.session_state.specialty = None

    file_type = st.radio("Select file type:", ("Image", "PDF"))

    if file_type == "Image":
        uploaded_file = st.file_uploader("Choose a medical report image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            image = Image.open(path)
            st.image(image, caption="Uploaded Medical Report", use_container_width=True)
            if st.button("Analyze Image Report"):
                st.session_state.analysis = analyze_medical_report(image, "image")
            os.unlink(path)
    else:
        uploaded_file = st.file_uploader("Choose a medical report PDF", type=["pdf"])
        if uploaded_file:
            st.write("PDF uploaded successfully")
            if st.button("Analyze PDF Report"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                with open(path, 'rb') as f:
                    text = extract_text_from_pdf(f)
                st.session_state.analysis = analyze_medical_report(text, "text")
                os.unlink(path)

    if st.session_state.analysis:
        st.subheader("Analysis Results:")
        st.markdown(
            f"""
            <div style='border: 2px solid #ff1493; padding: 15px; border-radius: 10px; background-color: #ffe6e6; color: #ff1493;'>
                {st.session_state.analysis}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Diagrammatic Representation:")
        flow_text = generate_flowchart_from_analysis(st.session_state.analysis)
        flowchart = render_flowchart(flow_text)
        st.graphviz_chart(flowchart.source)

        if st.session_state.specialty is None:
            spec_resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert in medical report classification."},
                    {"role": "user", "content": f"What specialty does this analysis pertain to?\n\n{st.session_state.analysis}"}
                ]
            )
            st.session_state.specialty = spec_resp.choices[0].message.content.strip()

        st.subheader("Ask Questions About the Report:")
        question = st.text_input("Your question:")
        if question:
            answer = chat_with_report(question, st.session_state.analysis)
            st.session_state.chat_history.append((question, answer))
        for q, a in st.session_state.chat_history:
            st.markdown(f"""
            <div style='border: 1px solid #cccccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 10px; color: #ff1493;'>
                <strong>You:</strong> {q}
            </div>
            <div style='border: 1px solid #ff1493; padding: 10px; border-radius: 5px; background-color: #ffe6e6; margin-bottom: 10px; color: #ff1493;'>
                <strong>Assistant:</strong> {a}
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Find Specialists")
        location = st.text_input("Enter location (city, area):")
        if location and st.button("Find Doctors"):
            doctors = find_doctors(st.session_state.specialty, location)
            st.write(doctors)

if __name__ == "__main__":
    main()
