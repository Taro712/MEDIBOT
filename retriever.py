from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# Get the HuggingFace API token (still needed for embeddings)
Hf_Token = os.getenv("Hf_TOKEN")
if not Hf_Token:
    raise ValueError("Please set the Hf_TOKEN environment variable")


# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'}
)

# Load vector store
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def search_documents(query):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever.invoke(query)

def output_documents(query):
    results = search_documents(query)
    documents = [doc.page_content for doc in results]
    return documents

#streamlit app

st.title("WELCOME TO MEDIBOT")
st.write("This is a simple app to search through medical documents.")
query = st.text_input("Enter your query here:")
if query:
    docs = output_documents(query)
    if docs:
        for i, doc in enumerate(docs):
            st.subheader(f"Document {i+1}")
            st.write(doc)
    else:
        st.warning("No documents found for the given query.")
else:
    st.info("Please enter a query to search for documents.")