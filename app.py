import streamlit as st
import google.generativeai as genai
from groq import Groq
from PIL import Image
import PyPDF2
import tempfile
import os
from google.api_core import exceptions
import time

# Hardcoded API keys
GEMINI_API_KEY = "AIzaSyDphqA2q_ae9YAHiElrPX96ULfLFtvbfpo"
GROQ_API_KEY = "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"

def configure_clients():
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    # Configure Groq client
    global groq_client
    groq_client = Groq(api_key=GROQ_API_KEY)

configure_clients()

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


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
                st.warning(f"An error occurred. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report after {MAX_RETRIES} attempts. Error: {str(e)}")
                return fallback_analysis(content, content_type)


def fallback_analysis(content, content_type):
    st.warning("Using fallback analysis method due to API issues.")
    if content_type == "image":
        return "Unable to analyze the image due to API issues. Please try again later or consult a medical professional for accurate interpretation."
    else:
        word_count = len(content.split())
        return f"""
        Fallback Analysis:
        1. Document Type: Text-based medical report
        2. Word Count: Approximately {word_count} words
        3. Content: The document appears to contain medical information, but detailed analysis is unavailable due to technical issues.
        4. Recommendation: Please review the document manually or consult with a healthcare professional for accurate interpretation.
        5. Note: This is a simplified analysis due to temporary unavailability of the AI service. For a comprehensive analysis, please try again later.
        """


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def chat_with_report(question, analysis_text):
    # Use Groq for follow-up chat
    messages = [
        {"role": "system", "content": "You are a knowledgeable medical assistant."},
        {"role": "system", "content": f"Report Analysis:\n{analysis_text}"},
        {"role": "user", "content": question}
    ]
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # adjust model as needed
        messages=messages
    )
    return resp.choices[0].message.content


def find_doctors(specialty, location):
    # Ask Groq to list top doctors for a specialty in a location
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


def main():
    st.title("AI-driven Medical Report Analyzer & Specialist Finder")
    st.write("Upload a medical report (image or PDF), get analysis, ask questions, and find relevant specialists.")

    # Session state
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'specialty' not in st.session_state:
        st.session_state.specialty = None

    file_type = st.radio("Select file type:", ("Image", "PDF"))

    # Upload & analyze
    if file_type == "Image":
        uploaded_file = st.file_uploader("Choose a medical report image", type=["jpg", "jpeg", "png"]);
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            image = Image.open(path)
            st.image(image, caption="Uploaded Medical Report", use_column_width=True)
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

    # Display analysis and derive specialty
    if st.session_state.analysis:
        st.subheader("Analysis Results:")
        st.write(st.session_state.analysis)
        # Let model extract specialty
        if st.session_state.specialty is None:
            spec_resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":"You are an expert in medical report classification."},
                    {"role":"user","content":f"Based on this analysis, what medical specialty does it pertain to?\nAnalysis:\n{st.session_state.analysis}"}
                ]
            )
            st.session_state.specialty = spec_resp.choices[0].message.content.strip()

        # Chatbot interface
        st.subheader("Ask Questions About the Report:")
        question = st.text_input("Your question:")
        if question:
            answer = chat_with_report(question, st.session_state.analysis)
            st.session_state.chat_history.append((question, answer))
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")

        # Specialist finder
        st.subheader("Find Specialists")
        location = st.text_input("Enter location (city, area):")
        if location and st.button("Find Doctors"):
            doctors = find_doctors(st.session_state.specialty, location)
            st.write(doctors)

if __name__ == "__main__":
    main()
