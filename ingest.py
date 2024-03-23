import os
import pickle
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from text import TextLoader

# Load environment variables
load_dotenv()

# Constants
LLAMAPARSE_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VECTOR_DB_FOLDER_NAME = "VectorDB"
DATA_FOLDER = os.path.join(os.getcwd(), "data")
VECTOR_DB_PATH = os.path.join(os.getcwd(), VECTOR_DB_FOLDER_NAME)

# Ensure VECTOR_DB_PATH exists
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def load_or_parse_data():
    """
    Load or parse data, depending on the availability of a pre-existing parsed file.
    """
    data_file = os.path.join(VECTOR_DB_PATH, 'parsed_data.pkl')
    
    if os.path.exists(data_file):
        with open(data_file, "rb") as file:
            return pickle.load(file)

    parsing_instruction = ("""The provided document contains information and guide about real estate in Swiss and other real estae information
                        If the document contain a benefits TABLE that describe coverage amounts, do not ouput it as a table, but instead as a list of benefits string.
                        Try to be precise while answering the questions.
                           """)
    parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown", parsing_instruction=parsing_instruction)

    llama_parse_documents = []
    for pdf_file in [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]:
        pdf_path = os.path.join(DATA_FOLDER, pdf_file)
        llama_parse_documents.extend(parser.load_data(pdf_path))

    with open(data_file, "wb") as file:
        pickle.dump(llama_parse_documents, file)
    
    return llama_parse_documents

def create_vector_database():
    """
    Create a vector database by loading documents, splitting them into chunks,
    transforming into embeddings, and persisting into a Qdrant vector database.
    """
    llama_parse_documents = load_or_parse_data()
    markdown_file_path = os.path.join(VECTOR_DB_PATH, 'processed_documents.md')
    
    # Write to the markdown file only if it doesn't exist to avoid appending to an existing file
    if not os.path.exists(markdown_file_path):
        with open(markdown_file_path, 'a', encoding='utf-8') as file:
            for document in llama_parse_documents:
                file.write(document.text + '\n')
                
    

    loader = DirectoryLoader(VECTOR_DB_PATH, glob="**/*.md", show_progress=True, loader_cls=TextLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    qdrant = Qdrant.from_documents(docs, embeddings, url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY, collection_name="testing")
    
    print('Vector DB created successfully!')

if __name__ == "__main__":
    create_vector_database()
