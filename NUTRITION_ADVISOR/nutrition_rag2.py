from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load prebuilt vectorstore
def query_nutrition_advice(query: str):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="NUTRITION_ADVISOR/vector_store",
        embedding_function=embedding_model,
    )

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = Ollama(model="llama3")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    result = chain.invoke({"query": query})
    return result
