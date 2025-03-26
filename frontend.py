import streamlit as st
import requests

# FastAPI Backend URL
FASTAPI_URL = "http://127.0.0.1:8000"

st.title("📄 PDF Processing & Q/A Service")

# Sidebar navigation
option = st.sidebar.radio("Choose an action:", ["Upload & Parse PDF", "Parse GCS PDF","Select chunking method","Select chunked output file","Select embedded output file","PineconeDB Indexing","ChromaDB Indexing","PineCone:Ask a Question","ChromaDB:Ask a Question","Ask a Research Question", "View Reports"])


# ✅ Upload & Parse a PDF
if option == "Upload & Parse PDF":
    st.subheader("📤 Upload a PDF File for Parsing")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    pdf_url = st.text_input("Or provide a URL to a PDF", "")

    # Select the parsing method
    parse_method = st.selectbox(
        "Select Parsing Method",
        ["pymupdf","mistral", "docling"],
        index=0
    )

    if uploaded_file is not None or pdf_url:
        if st.button("🚀 Upload & Parse"):
            # Prepare the data to send based on the method
            files = {}
            if uploaded_file is not None:
                file_name = uploaded_file.name  # Retain the original file name
                files = {
                    "file": (file_name, uploaded_file.getvalue(), "application/pdf")
                }
            
            # Send a request to FastAPI with the selected method and either file or URL
            if uploaded_file:
                response = requests.post(
                    f"{FASTAPI_URL}/upload_and_parse_pdf/?parse_method={parse_method}",
                    files=files
                )
            else:  # If no file, send URL
                response = requests.post(
                    f"{FASTAPI_URL}/process-pdf/",
                    json={"pdf_url": pdf_url}  # Send the PDF URL to the backend
                )

# ✅ Parse a Selected PDF from GCS
elif option == "Parse GCS PDF":
    st.subheader("📜 Select a PDF from GCS for Parsing")

    # Fetch the list of files from the FastAPI endpoint
    response = requests.get(f"{FASTAPI_URL}/list_pdf_files")
    
    if response.status_code == 200:
        files = response.json().get("files", [])
        
        if files:
            # Let the user select a file from the list
            selected_file = st.selectbox("Choose a PDF file:", files)

            # Select the parsing method
            parse_method = st.selectbox(
                "Select Parsing Method",
                ["pymupdf", "docling"],
                index=0
            )

            if selected_file and st.button("🚀 Parse Selected PDF"):
                # Request to parse the selected file from GCS with the selected parse method
                response = requests.get(
                    f"{FASTAPI_URL}/parse_gcs_pdf",
                    params={"file_name": selected_file, "parse_method": parse_method}
                )

                if response.status_code == 200:
                    # If parsing is successful, display the extracted markdown content
                    markdown_content = response.json().get("markdown_content", "")
                    st.success(f"✅ File **{selected_file}** parsed successfully using **{parse_method}**!")
                    st.subheader("📜 Extracted Markdown Content:")
                    st.markdown(markdown_content)
                else:
                    # Show error if something went wrong with parsing
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.warning("❌ No PDF files available for parsing.")
    else:
        # Show an error if the list of PDFs cannot be fetched
        st.error("❌ Failed to fetch PDF list.")

# Your existing logic to select the chunking method
elif option == "Select chunking method":
    st.subheader("📜 Select an extracted PDF from GCS for chunking")

    # Fetch the list of extracted files from the backend
    response = requests.get(f"{FASTAPI_URL}/list_extracted_files")

    if response.status_code == 200:
        files = response.json().get("files", [])
        if files:
            # Dropdown to select a file
            selected_file = st.selectbox("Choose a file:", files)

            # Dropdown to select chunking strategy
            strategy = st.selectbox("Select chunking strategy:", ["fixed", "sentence", "sliding", "recursive","langchain"])

            # Process file button
            if st.button("Process File"):
                with st.spinner("Processing..."):
                    fetch_response = requests.get(
                        f"{FASTAPI_URL}/fetch_file/",
                        params={"file_name": selected_file, "strategy": strategy}
                    )

                    if fetch_response.status_code == 200:
                        st.success(f"✅ File '{selected_file}' processed successfully with {strategy} chunking!")
                    else:
                        st.error(f"❌ Error: {fetch_response.json().get('detail', 'Unknown error')}")
        else:
            st.warning("No files found in GCS.")
    else:
        st.error("Failed to fetch extracted files.")

elif option == "Select chunked output file":
    st.subheader("📂 Select a Chunked Output File")

    # Fetch the list of chunked files from the backend
    response = requests.get(f"{FASTAPI_URL}/list_chunked_output_files")

    if response.status_code == 200:
        files = response.json().get("files", [])

        if files:
            # Dropdown to select a chunked file
            selected_file = st.selectbox("Choose a file:", files)

            # Button to fetch file content and trigger embeddings
            if st.button("🔍 Fetch & Generate Embeddings"):
                with st.spinner(f"Fetching content and generating embeddings for '{selected_file}'..."):
                    # Fetch file content and initiate embedding generation
                    fetch_response = requests.get(
                        f"{FASTAPI_URL}/fetch_file_content",
                        params={"file_name": selected_file}
                    )

                    if fetch_response.status_code == 200:
                        # Display the embedding initiation message
                        st.success(f"✅ Embedding generation initiated for '{selected_file}'!")

                        # Display file name and status
                        file_name = fetch_response.json().get("file_name", "")
                        status = fetch_response.json().get("status", "")
                        st.write(f"**File:** {file_name}")
                        st.write(f"**Status:** {status}")
                    
                    else:
                        st.error(f"❌ Error: {fetch_response.json().get('detail', 'Unknown error')}")
        else:
            st.warning("No chunked files found.")
    else:
        st.error("Failed to fetch chunked files.")

elif option == "Select embedded output file":
    st.subheader("📂 Select Embedded Output File")

    # Fetch the list of embedded files from the backend
    response = requests.get(f"{FASTAPI_URL}/list_embedded_output_files")

    if response.status_code == 200:
        files = response.json().get("files", [])
        
        if files:
            # Dropdown to select an embedded file
            selected_file = st.selectbox("Choose an embedded file:", files)

            # Text input for the search query
            query = st.text_input("🔍 Enter your search query:", "")

            # Optional quarter filter
            quarter_filter = st.text_input("📅 Enter quarter filter (optional):", "")

            # Number of top results to fetch
            top_n = st.slider("🔢 Number of top results:", min_value=1, max_value=10, value=5)

            # Button to fetch and search the file content
            if st.button("📜 Fetch & Search Embedded File Content"):
                if not query.strip():
                    st.warning("⚠️ Please enter a search query.")
                else:
                    with st.spinner(f"Fetching and searching in '{selected_file}'..."):
                        
                        # Fetch content of the selected file with the query
                        fetch_response = requests.get(
                            f"{FASTAPI_URL}/fetch_embedded_file_content",
                            params={
                                "file_name": selected_file,
                                "query": query,
                                "quarter_filter": quarter_filter if quarter_filter.strip() else None,
                                "top_n": top_n
                            }
                        )

                        if fetch_response.status_code == 200:
                            file_name = fetch_response.json().get("file_name", "")
                            search_results = fetch_response.json().get("results", [])
                            gpt_response = fetch_response.json().get("gpt_response", "")

                            st.success(f"✅ File '{file_name}' searched successfully!")
                            st.subheader("🔍 Search Results:")

                            # Display search results
                            if search_results:
                                for idx, result in enumerate(search_results, start=1):
                                    st.subheader(f"📄 Result {idx}")
                                    st.write(f"**Similarity Score:** {round(result['similarity'], 4)}")
                                    st.write(f"**Text Chunk:**\n{result['chunk']}\n")
                            else:
                                st.warning("❌ No matching results found.")

                            # Display GPT response
                            st.subheader("🤖 GPT-40-mini Response:")
                            st.write(gpt_response if gpt_response else "❌ No response generated.")
                        
                        else:
                            st.error(f"❌ Error: {fetch_response.json().get('detail', 'Unknown error')}")
        else:
            st.warning("⚠️ No embedded files found.")
    else:
        st.error("❌ Failed to fetch embedded files.")

elif option == "PineconeDB Indexing":
    st.subheader("📂 Select a JSON file from GCS for indexing")
    
    # Fetch the list of extracted files from the backend
    response = requests.get(f"{FASTAPI_URL}/list_chunked_output_files")
    
    if response.status_code == 200:
        files = response.json().get("files", [])
        if files:
            # Dropdown to select a file
            selected_file = st.selectbox("Choose a file:", files)
            
            # Process file button
            if st.button("Index File"):
                with st.spinner("Indexing..."):
                    index_response = requests.post(
                        f"{FASTAPI_URL}/index-json/",
                        data={"file_path": selected_file}
                    )
                    
                    if index_response.status_code == 200:
                        st.success(f"✅ File '{selected_file}' successfully indexed!")
                    else:
                        st.error(f"❌ Error: {index_response.json().get('detail', 'Unknown error')}")
        else:
            st.warning("No files found in GCS.")
    elif response.status_code != 200:
        st.error("Failed to fetch extracted files.")

elif option == "ChromaDB Indexing":
    st.subheader("📂 Select a JSON file from GCS for indexing")
    
    # Fetch the list of extracted files from the backend
    response = requests.get(f"{FASTAPI_URL}/list_chunked_output_files")
    
    if response.status_code == 200:
        files = response.json().get("files", [])
        if files:
            # Dropdown to select a file
            selected_file = st.selectbox("Choose a file:", files)
            
            # Process file button
            if st.button("Index File"):
                with st.spinner("Indexing..."):
                    index_response = requests.post(
                        f"{FASTAPI_URL}/index-json-chroma/",
                        data={"file_path": selected_file}
                    )
                    
                    if index_response.status_code == 200:
                        st.success(f"✅ File '{selected_file}' successfully indexed!")
                    else:
                        st.error(f"❌ Error: {index_response.json().get('detail', 'Unknown error')}")
        else:
            st.warning("No files found in GCS.")
    elif response.status_code != 200:
        st.error("Failed to fetch extracted files.")

elif option == "PineCone:Ask a Question":
    st.subheader("🤖 Ask a Question About Your PDFs")

    query = st.text_input("Enter your question:")

    if query:
        if st.button("🔍 Ask"):
            try:
                # Send query as a URL parameter
                response = requests.post(f"{FASTAPI_URL}/ask", params={"query": query})

                if response.status_code == 200:
                    result = response.json().get("response", "")
                    st.success("✅ Response Retrieved!")
                    st.subheader("💬 Answer:")
                    st.write(result)
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Exception: {str(e)}")

elif option == "ChromaDB:Ask a Question":
    st.subheader("🤖 Ask a Question About Your PDFs")

    query = st.text_input("Enter your question:")

    if query:
        if st.button("🔍 Ask"):
            try:
                # Send query as a URL parameter
                response = requests.post(f"{FASTAPI_URL}/ask-chromadb", params={"query": query})

                if response.status_code == 200:
                    result = response.json().get("response", "")
                    st.success("✅ Response Retrieved!")
                    st.subheader("💬 Answer:")
                    st.write(result)
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Exception: {str(e)}")

# ✅ Ask a Research Question
elif option == "Ask a Research Question":
    st.subheader("❓ Research Query Input")

    # User input for query
    query = st.text_area("Enter your research question about NVIDIA:", "")

    if st.button("🚀 Get Insights"):
        if not query.strip():
            st.warning("Please enter a research question before proceeding.")
        else:
            # Send request to FastAPI backend
            response = requests.post(
                f"{FASTAPI_URL}/ask_question",
                json={"query": query}
            )

            if response.status_code == 200:
                result = response.json()
                st.subheader("📑 Research Report")
                st.write(f"**Query:** {result['query']}")

                # Display real-time insights (Web Search results)
                if "web_results" in result and result["web_results"]:
                    st.subheader("🌍 Latest Web Insights")
                    for news in result["web_results"]:
                        st.markdown(f"🔗 [{news['title']}]({news['link']})")
                else:
                    st.write("No recent news found for this query.")

            else:
                st.error("Failed to fetch insights. Please try again.")