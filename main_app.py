import streamlit as st
import requests
import json

st.title("AI Candidate Answer Evaluation App")

BACKEND_URL = "http://localhost:8000/transcribe"

uploaded = st.file_uploader("Upload MP3 / WAV file", type=["mp3", "wav", "m4a"])

if uploaded:
    st.audio(uploaded)

    if st.button("Transcribe & Evaluate"):
        files = {"file": uploaded.getvalue()}
        with st.spinner("Processing..."):
            try:
                response = requests.post(BACKEND_URL, files=files)
                result = response.json()

                st.subheader("Transcript")
                st.write(result["transcript"])

                st.subheader("Extracted Q&A")
                st.json(result["qa_extracted"])

            except Exception as e:
                st.error(f"Failed to call backend: {e}")
