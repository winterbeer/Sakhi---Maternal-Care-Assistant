from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.llms import Ollama

def custom_loader(file_path: Path):
    try:
        
        return PyPDFLoader(str(file_path))
    except Exception as e:
        print(f"Standard PDF parse failed for {file_path}, using OCR: {e}")
        
        return UnstructuredPDFLoader(str(file_path), strategy="hi_res")



loader = DirectoryLoader(
    path = "NUTRITION_ADVISOR",
    glob="*.pdf",
    loader_cls=custom_loader,
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
all_chunks = []
for doc in docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_chunks.extend(chunks)


#embedding chunks
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_texts(
    texts=all_chunks,
    embedding=embedding_model,
    persist_directory="NUTRITION_ADVISOR/vector_store",
)

vector_store.get(include=["metadatas", "documents", "embeddings"])

#search the docs

from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

llm = Ollama(model="llama3")


