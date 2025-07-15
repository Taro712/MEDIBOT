from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv



load_dotenv()


DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"
 
def load_data(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(text_chunks):
    embeddings = None
    
    try:
        # Try using OpenAI embeddings first
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        print("Using OpenAI embeddings")
    except Exception as e:
        print(f"OpenAI embeddings failed: {e}")
        try:
            # Fallback to HuggingFace embeddings
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("Using HuggingFace embeddings")
        except Exception as e:
            print(f"HuggingFace embeddings failed: {e}")
            print("Both embedding models failed")
            return None  # or raise an exception
    
    # Only proceed if we have valid embeddings
    if embeddings is not None:
        try:
            vector_store = FAISS.from_documents(text_chunks, embeddings)
            vector_store.save_local(DB_FAISS_PATH)
            return vector_store
        except Exception as e:
            print(f"Failed to create vector store: {e}")
            return None
    else:
        print("No embeddings available - cannot create vector store")
        return None
    

doc = load_data(DATA_PATH)
text_chunks = create_chunks(doc)
create_vector_store(text_chunks)
