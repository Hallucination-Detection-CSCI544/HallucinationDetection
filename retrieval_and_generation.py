from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline # Updated import
from langchain.chains import RetrievalQA
from transformers import pipeline
import torch

class RAGPipeline:
    def __init__(self, model, tokenizer, embeddings_model_name, vectorstore_folder="rag_triviaqa_store", top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k

        # Embeddings + FAISS
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        self.vectorstore = FAISS.load_local(
            vectorstore_folder, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # LLM pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024, 
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

    def query(self, text: str):
        result = self.qa.invoke({"query": text})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
