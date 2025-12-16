from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RAGPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        embeddings_model_name,
        vectorstore_folder="rag_triviaqa_store",
        top_k=5,
        max_new_tokens=10,
        temperature=0.0,
        do_sample=False,
        top_p=1.0
    ):
        self.top_k = top_k

        # Embeddings + FAISS
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model_name
        )

        self.vectorstore = FAISS.load_local(
            vectorstore_folder,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # HF generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            return_full_text=False,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Prompt
        self.prompt = ChatPromptTemplate.from_template("answer according to the context in one or two words. context: {context}\nquestion: {input}\nanswer:")

        # LCEL RAG chain
        self.chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "input": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
        )

    def query(self, text: str):
        answer = self.chain.invoke(text)

        docs = self.retriever.invoke(text)
        doc_txt = "\n\n".join(doc.page_content for doc in docs)
        print(f"Docs retrieved: {doc_txt}")
        return {
            "answer": str(answer),
            "sources": docs
        }

