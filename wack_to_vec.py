import os
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def build_rag_store(
        n: int = 1000,
        output_folder: str = "rag_triviaqa_store",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
        ) -> None:

    if os.path.exists(output_folder):
        print(f"Vector store already exists at {output_folder}. Skipping build.")
        return

    # -----------------------
    # 1. Load TriviaQA (rc)
    # -----------------------
    print("Loading TriviaQA rc...")
    try:
        dataset = load_dataset("trivia_qa", "rc", split=f"validation[:{n}]", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    rag_docs = []

    # -----------------------
    # 2. Initialize text splitter
    # -----------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len
    )

    # -----------------------
    # 3. Extract RAG documents
    # -----------------------
    print("Extracting RAG documents...")

    for q_idx, row in enumerate(dataset):
        question_id = q_idx

        entity_pages = row.get("entity_pages", {})
        search_results = row.get("search_results", {})

        # --- Wikipedia pages ---
        ep_contexts = entity_pages.get("wiki_context", [])
        ep_titles = entity_pages.get("title", [])
        ep_urls = entity_pages.get("url", [])

        for i, text in enumerate(ep_contexts):
            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "question_id": question_id,
                        "title": ep_titles[i] if i < len(ep_titles) else "Unknown",
                        "url": ep_urls[i] if i < len(ep_urls) else "Unknown",
                        "source": "wiki"
                    }
                )
                # Split long pages into chunks
                chunks = splitter.split_documents([doc])
                for idx, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = idx
                    rag_docs.append(chunk)

        # --- Web search result snippets ---
        sr_contexts = search_results.get("search_context", [])
        sr_titles = search_results.get("title", [])
        sr_urls = search_results.get("url", [])

        for i, text in enumerate(sr_contexts):
            if text and text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "question_id": question_id,
                        "title": sr_titles[i] if i < len(sr_titles) else "Unknown",
                        "url": sr_urls[i] if i < len(sr_urls) else "Unknown",
                        "source": "web"
                    }
                )
                chunks = splitter.split_documents([doc])
                for idx, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = idx
                    rag_docs.append(chunk)

    print(f"Created {len(rag_docs)} RAG documents.")

    # -----------------------
    # 4. Build Vector Store
    # -----------------------
    print("Embedding and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if rag_docs:
        vectorstore = FAISS.from_documents(rag_docs, embeddings)
        vectorstore.save_local(output_folder)
        print(f"Saved RAG datastore to '{output_folder}'.")
    else:
        print("No documents found to index.")
