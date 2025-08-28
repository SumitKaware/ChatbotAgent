import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import the text splitter
from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument # Alias to avoid conflict if needed
from langchain_community.vectorstores import Chroma
from pydantic import SecretStr
from langchain_community.tools.tavily_search import TavilySearchResults
from langfuse import Langfuse

client = Langfuse()
print(client)

# Load environment variables from .env file
load_dotenv()

# --- 1. Configuration and Setup ---
#
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google API Key or set it as an environment variable.
# You can get an API key from Google AI Studio: https://aistudio.google.com/app/apikey
# It's recommended to set it as an environment variable: export GOOGLE_API_KEY="your_api_key_here"
# If running in an environment where __api_key__ is provided (e.g., Canvas), it will be used.
# api_key = os.getenv("GOOGLE_API_KEY", "")
# tavily_api_key = os.getenv("TAVILY_API_KEY", "TAVILY_API_KEY")
# if not api_key or not tavily_api_key:
#     # Fallback for Canvas environment if __api_key__ is not set as GOOGLE_API_KEY
#     # In a real application, you'd handle this more robustly (e.g., prompt user, raise error).
#     print("Warning: GOOGLE_API_KEY environment variable not set. Please set it or provide it directly.")
#     # Attempt to use a global variable if available, common in some interactive environments
#     api_key = globals().get('__api_key__', '')
#     if not api_key:
#         raise ValueError("Google API Key is not set. Please set GOOGLE_API_KEY environment variable.")
# # from google.colab import userdata
# # api_key = userdata.get('GOOGLE_API_KEY_1')

# # Initialize the Generative AI model
# # We'll use gemini-2.0-flash for faster responses.
# import os

api_key = os.getenv("GOOGLE_API_KEY", "")
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
langfuse_host = os.getenv("LANGFUSE_HOST", "")

if not api_key or not tavily_api_key or not langfuse_secret_key or not langfuse_public_key or not langfuse_host:
    print("Warning: API keys not set. Please set GOOGLE_API_KEY, TAVILY_API_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_HOST environment variables.")

    # Fallback for interactive notebooks / environments
    api_key = globals().get('__api_key__', api_key)
    tavily_api_key = globals().get('__tavily_api_key__', tavily_api_key)
    langfuse_secret_key = globals().get('__langfuse_secret_key__', langfuse_secret_key)
    langfuse_public_key = globals().get('__langfuse_public_key__', langfuse_public_key)
    langfuse_host = globals().get('__langfuse_host__', langfuse_host)

    if not api_key:
        raise ValueError("Google API Key is not set. Please set GOOGLE_API_KEY environment variable.")
    if not tavily_api_key:
        raise ValueError("Tavily API Key is not set. Please set TAVILY_API_KEY environment variable.")



langfuse = Langfuse(
  secret_key="sk-lf-4e091f50-f7ac-4833-9478-6739c43b9818",
  public_key="pk-lf-d4a90d9a-8541-4f52-8f21-faf596e27f95",
  host="https://us.cloud.langfuse.com"
)



llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.2, top_p=0.95)

# Initialize the embedding model for ChromaDB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(api_key))

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

tavily_search = TavilySearchResults(
    max_results=3,
    topic="general",
    include_answer=True,
    include_raw_content=False,
    search_depth="basic",
)

def vector_db_setup(documents: List[LangchainDocument], persist_directory: str = "chroma_db"):
    """
    Initializes a Chroma vector store from the provided documents.

    Args:
        documents (List[LangchainDocument]): A list of LangchainDocument objects to be stored.
        persist_directory (str): The directory where the ChromaDB instance will be saved.
    """
    print("Initializing ChromaDB with sample documents...")
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    print("ChromaDB initialized.")
    return vectorstore


