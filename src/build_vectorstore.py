import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load your API key
load_dotenv()
embedding = OpenAIEmbeddings()

# Load documents
loader = DirectoryLoader("reddit-posts", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Build vector store
vectordb = Chroma.from_documents(chunks, embedding, persist_directory="vectorstore")

vectordb.persist()
print(f"Vectorstore saved with {len(chunks)} chunks.")
