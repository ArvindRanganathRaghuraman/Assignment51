from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.mlops import Mlflow
from diagrams.custom import Custom
from diagrams.gcp.storage import GCS
from diagrams.programming.language import Python
from diagrams.programming.flowchart import Database
from diagrams.generic.compute import Rack

with Diagram("RAG Pipelines and Airflow Architecture", filename="diagrams/rag_architecture", show=False, direction="TB"):

    user = User("User")

    # =========================
    # AIRFLOW PIPELINE 1 BLOCK
    # =========================
    with Cluster("Airflow Pipeline 1: Docling - Markdown Extraction"):
        airflow1 = Airflow("Airflow DAG 1")
        scrape_pdfs = Python("Scrape NVIDIA PDFs\n(10K/10Q - 5 Years)")
        gcs_bucket_1 = GCS("GCS Bucket\n(PDF Storage)")
        docling = Custom("Docling\n(Generate Markdown)", "/Users/kaushikj/Desktop/Diagrams/Docling.png")
        markdown_output = Rack("Markdown Output")

        airflow1 >> scrape_pdfs >> gcs_bucket_1 >> docling >> markdown_output

    # =========================
    # AIRFLOW PIPELINE 2 BLOCK
    # =========================
    with Cluster("Airflow Pipeline 2: Mistral OCR - Text Extraction"):
        airflow2 = Airflow("Airflow DAG 2")
        scrape_pdf_links = Python("Scrape PDF Links")
        gcs_bucket_2 = GCS("GCS Bucket\n(PDF Link Folder)")
        mistral_ocr = Custom("Mistral OCR", "/Users/kaushikj/Desktop/Diagrams/Mistral.png")
        advanced_text_output = Rack("Advanced Text Output")

        airflow2 >> scrape_pdf_links >> gcs_bucket_2 >> mistral_ocr >> advanced_text_output

    # =========================
    # RAG PIPELINE 1 (Local)
    # =========================
    with Cluster("RAG Pipeline 1: Manual (No Vector DB)"):
        chunking_manual = Python("Chunking\n(Local)")
        embeddings_manual = Python("Embeddings")
        cosine_similarity = Python("Cosine Similarity\nSearch Logic")
        local_chunks = Database("Chunks (Local)")
        not_relevant = Rack('"Not Relevant Context" Message')

        advanced_text_output >> chunking_manual >> embeddings_manual >> local_chunks
        user >> cosine_similarity >> local_chunks >> not_relevant

    # =========================
    # RAG PIPELINE 2 (Pinecone)
    # =========================
    with Cluster("RAG Pipeline 2: Pinecone Vector DB"):
        chunking_pinecone = Python("Chunking")
        embeddings_pinecone = Python("Embeddings")
        pinecone_index = Custom("Pinecone DB", "/Users/kaushikj/Desktop/Diagrams/Pinecone.png")
        not_relevant_2 = Rack('"Not Relevant Context" Message')

        markdown_output >> chunking_pinecone >> embeddings_pinecone >> pinecone_index
        user >> pinecone_index >> not_relevant_2

    # =========================
    # RAG PIPELINE 3 (ChromaDB)
    # =========================
    with Cluster("RAG Pipeline 3: ChromaDB"):
        chunking_chroma = Python("Chunking")
        embeddings_chroma = Python("Embeddings")
        chroma_index = Custom("ChromaDB", "/Users/kaushikj/Desktop/Diagrams/chromadb.png")
        not_relevant_3 = Rack('"Not Relevant Context" Message')

        markdown_output >> chunking_chroma >> embeddings_chroma >> chroma_index
        user >> chroma_index >> not_relevant_3

    # =========================
    # INTEGRATION LAYER
    # =========================
    with Cluster("Backend + Frontend"):
        fastapi = Python("FastAPI (All Pipelines)")
        streamlit_ui = Custom("Streamlit App", "/Users/kaushikj/Desktop/Diagrams/streamlit 1.png")
        docker_backend = Custom("Docker (FastAPI)", "/Users/kaushikj/Desktop/Diagrams/docker.png")
        docker_frontend = Custom("Streamlit Cloud", "/Users/kaushikj/Desktop/Diagrams/streamlit_cloud.png")

        user >> streamlit_ui >> fastapi
        fastapi >> [cosine_similarity, pinecone_index, chroma_index]
        fastapi >> [docker_backend, docker_frontend]

    # =========================
    # LLM MODELS
    # =========================
    with Cluster("LLM Engines"):
        gpt4 = Custom("ChatGPT 4-o", "/Users/kaushikj/Desktop/Diagrams/ChatGPT.png")
        gemini = Custom("Google Gemini", "/Users/kaushikj/Desktop/Diagrams/Gemini.png")

        fastapi >> [gpt4, gemini]