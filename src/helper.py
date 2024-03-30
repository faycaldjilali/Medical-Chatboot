



from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Access the API key using os.environ
cohere_api_key = os.environ.get("COHERE_API_KEY")



#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def coher_embeddings():
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)

    return embeddings