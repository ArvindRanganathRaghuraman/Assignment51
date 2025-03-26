import streamlit as st
import requests

# FastAPI backend URL
BASE_URL = "http://127.0.0.1:8000"

st.title("📄 PDF to Markdown Converter, Summarizer & Q/A")

# List files in GCS
st.header("Available Markdown Files")
list_response = requests.get(f"{BASE_URL}/list_files/")
if list_response.status_code == 200:
    files_list = list_response.json().get("files", [])
    selected_file = st.selectbox("Select a file", files_list)
else:
    st.error("Failed to fetch files from storage.")

# View file content
file_content = None
if st.button("View File Content"):
    if selected_file:
        file_response = requests.get(f"{BASE_URL}/get_file/{selected_file}")
        if file_response.status_code == 200:
            file_data = file_response.json()
            file_content = file_data.get("content", "")

            # ✅ Ensure the content is not empty before displaying
            if file_content:
                st.subheader("File Content")
                st.text_area("Markdown Content", file_content, height=300)

                # ✅ Store file content in session state for Q/A and Summarization
                st.session_state["selected_file"] = selected_file
                st.session_state["file_content"] = file_content
            else:
                st.error("File content is empty.")
        else:
            st.error("Failed to fetch file content.")

# Q/A Section
st.header("Ask a Question (Based on Markdown)")
user_question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if "file_content" in st.session_state and st.session_state["file_content"]:
        qa_payload = {"question": user_question}
        qa_response = requests.post(f"{BASE_URL}/ask_question/", json=qa_payload)

        if qa_response.status_code == 200:
            st.subheader("Answer")
            st.text_area("Response", qa_response.json().get("answer"), height=150)
        else:
            st.error("Failed to get an answer.")
    else:
        st.warning("Please view a file first before asking a question.")

# Summarization
st.header("Summarization")
model_options = ["gpt", "gemini", "deepseek", "claude"]
selected_model = st.selectbox("Choose a Summarization Model", model_options)

if st.button("Summarize File"):
    if "file_content" in st.session_state and st.session_state["file_content"]:
        summarize_payload = {"content": st.session_state["file_content"], "model": selected_model}
        summarize_response = requests.post(f"{BASE_URL}/summarize_file/", json=summarize_payload)

        if summarize_response.status_code == 200:
            st.subheader("Summarized Text")
            st.text_area("Summary", summarize_response.json().get("summary"), height=200)
        else:
            st.error("Failed to summarize file.")
