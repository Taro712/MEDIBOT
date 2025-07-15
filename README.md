# MediBot

A medical chatbot that can answer questions based on medical documents using LangChain and HuggingFace.

## Features

- PDF document loading and processing
- Vector store creation for efficient document retrieval
- Question answering using HuggingFace models
- Fallback between OpenAI and HuggingFace embeddings
- Streamlit web interface (commented out, can be enabled)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   Hf_TOKEN=your_huggingface_token
   OPENAI_API_KEY=your_openai_key  # Optional
   ```
4. Place your PDF documents in the `data` folder
5. Run the following commands:
   ```bash
   # First, create the vector store
   python create_memory.py
   
   # Then run the chat interface
   python retriever.py
   ```

## Project Structure

- `create_memory.py`: Creates vector store from PDF documents
-  retriever.py`: Implements the chat interface
- `data/`: Directory for PDF documents
- `vectorstore/`: Directory for FAISS vector store
