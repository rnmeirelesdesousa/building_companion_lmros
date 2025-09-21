import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import the centralized, absolute path from the config file
from src.config import VECTOR_STORE_DIR

class RAGAgent:
    def __init__(self):
        print("Initializing RAG Agent...")

        # 1. Load Embedding Model
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

        # 2. Load the Vector Store using the correct path from config
        print(f"Loading vector store from: {VECTOR_STORE_DIR}")
        if not os.path.exists(VECTOR_STORE_DIR):
            raise FileNotFoundError(
                f"Vector store not found at path: {VECTOR_STORE_DIR}. "
                "Please run the '01-Data-Ingestion-and-Embedding.ipynb' notebook first."
            )
        self.db = FAISS.load_local(
            VECTOR_STORE_DIR,
            self.embedder,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever()
        print("✅ Vector store loaded.")

        # 3. Initialize the LLM Client
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
        )

        # 4. Define the Prompt Template (using a default persona)
        prompt_template = """
        You are a helpful AI assistant for homeowners in Portugal.
        Use the provided legal information from the Regulamento Geral das Edificações Urbanas to answer the question
        in a simple, easy-to-understand way. Explain the key points without complex legal jargon.
        If you don't know the answer from the context, state that you do not know.

        Based on the regulations: {context}

        Question: {question}

        Helpful Answer (in Portuguese):
        """
        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # 5. Create the RetrievalQA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False, # We only need the final answer for the API
            chain_type_kwargs={"prompt": self.prompt}
        )
        print("✅ RAG Agent chain is ready.")


    def ask(self, question: str):
        """
        Asks a question to the RAG chain and returns the answer.
        """
        if not question:
            return "Please provide a question."

        try:
            # The .invoke method is standard for newer LangChain versions
            result = self.qa_chain.invoke({"query": question})
            return result.get('result', "No answer could be generated.")
        except Exception as e:
            print(f"❌ Error during RAG agent query: {e}")
            return "Sorry, I encountered an error while processing your request."