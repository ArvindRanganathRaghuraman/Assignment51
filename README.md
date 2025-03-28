# Assignment4-1
**Overview**
This is an extension to the first assignment wherein we use Large Language Models to Summarize the previously parsed document. Also, we have the feature to ask questions about the document. This is done by using RAG methodology.

**Contribution**

Kaushik - Deepseek, Claude Models (Summarization and QA), Diagrams.

Arvind - GPT-4o-mini, Gemini, Groq Models(Summarization and QA), Codelabs, Sample POCS for frontend and backend

Riya - Frontend, Backend and Deployment.

Codelabs - [https://docs.google.com/document/d/1MiDw0Wc_P5yULQMNc03unchqYNCEI25uHvXQqc18LRQ/edit?tab=t.0#heading=h.31ankfht5pc0
](https://codelabs-preview.appspot.com/?file_id=1MiDw0Wc_P5yULQMNc03unchqYNCEI25uHvXQqc18LRQ#3)


**Architecture Diagram**
![WhatsApp Image 2025-03-14 at 10 29 16 AM](https://github.com/user-attachments/assets/ba486e58-4c1d-4f50-874e-b2daed6b6b2a)

First, the user uploads a PDF and it is then parsed and converted to markdown.
It is then stored in google buckets.
We then create an API Endpoint pointing to the summarization engine, which lets you choose different models.
Finally, the endpoint is deployed on the cloud using docker.



**Folders.**
POC - Contains some samples of Gemini and Gpt Models and some frontend, and backend POCs
Backend - Contains the Backend and deployment files.
Frontend.py - the final frontend code.

# Assignment4-2
**Overview**

This is the continuation of the 4th Assignment part 1, where we have implemented multiple chunking strategies while using LLMs, when the user asks a question relevant to the context, the most relevant chunk will be selected, and if the user is asking a question that is not relevant to the context present in the database within PineCone and ChromaDb, the most irrelevant chunk will be selected and at the end will be mentioned as "Context not Available".

**Contributions**

Kaushik - Web scraping of NVIDIA website to get PDFs which are fed to Docling to generate Markowns and Web scraping of NVIDIA website to get PDF links which are fed to Mistral for advanced text Extraction. Building 2 Airflow DAGS for this process.

Arvind - Created the RAG Pipeline for Pinecone and Chromadb by creating indexing and embeddings and storing the chunks in PineCone DB and Chroma DB, where the user can ask questions using the LLMs ChatGPt and Google Gemini. The chunks with the best cosine similarity will be provided to the user depending on the context of the question the user is asking. Deployed the Backend code using Docker Compose and created a Docker Image.

Riya - Created a Manual RAG Pipeline with no Vector Database and the chunks are stored locally depending on the document we are using, here also we are using ChatGPt and Google Gemini as LLMs to respond by calling their API Keys. The chunks with the best cosine similarity will be taken depending on the context the user is asking for and the least relevant chunks will be given when the context of the chunks do not match with the question the user is asking. Also Integrated all the Airflow DAGS, storing the data in GCS Buckets, Creating RAGS for 3 different Pipelines using FASTAPI backend by creating multiple endpoints. Also, we have provided visualization using the Streamlit app, where the user can ask questions and select the chunking strategy and the LLMs he/she wants to use. The Streamlit part is deployed into the Streamlit Cloud.

**CodeLabs**

https://codelabs-preview.appspot.com/?file_id=1ZXzyDzWTK9nD04Mf43efhiRO9CnKPUpcgm4mFg_-Lz0/#0

**Google Docs**

https://docs.google.com/document/d/1ZXzyDzWTK9nD04Mf43efhiRO9CnKPUpcgm4mFg_-Lz0/edit?tab=t.0#heading=h.qytj5xmqxyaj

**Architecture Diagrams**

![rag_architecture](https://github.com/user-attachments/assets/75fb779e-bf0b-4819-bc3b-1a421fc9fef0)


# Assignment 5-1
**Overview**

In this assignment, we have continued from assignment 4, part 2, where we created 3 different agents to fetch NVIDIA data from different sources and create a report using all of this data.


**Architecture Diagram**

![image](https://github.com/user-attachments/assets/1c35fa84-15c9-4401-821c-3c25325d0950)


**Contributions**





